import pickle
# import tarfile
import pandas as pd
from catboost import CatBoostClassifier
from rank_bm25 import BM25Okapi

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class Submission():
    def __init__(self, test_features_df: pd.DataFrame, cat_level:str=1):
        self.test_df = test_features_df
        self.cat_level = cat_level

    def get_columns(self):
        self.target_column = 'target'
        # self.features_columns = self.test_df.drop(columns=['target'], errors='ignore').columns
        # self.features_columns = self.model_cat.feature_names_
        self.features_columns = list(self.model_cat.values())[0].feature_names_
        # self.cat_columns = list(np.array(self.features_columns)[list(self.model_cat.values())[0].get_cat_feature_indices()])

    def load_model(self, model_path: str = 'model.pkl'):
        # Сохраняем  модель
        with open(model_path, 'rb') as file:
            self.model_cat = pickle.load(file)
        self.get_columns()

    def predict(self):
        for cat in self.model_cat.keys():
            self.test_df.loc[self.test_df[f'cat_level_{self.cat_level}_1'] == cat, f'target_cat_{self.cat_level}'] = self.model_cat[cat].predict_proba(
                self.test_df[self.test_df[f'cat_level_{self.cat_level}_1'] == cat][self.features_columns])[:, 1]
        return self.test_df
        # self.test_df['target'] = self.model_cat.predict_proba(self.test_df[self.features_columns])[:, 1]

    # def save_submission(self, submission_file: str = 'data/submission.csv'):
    #     self.submission = self.test_df[['variantid1', 'variantid2', 'target']]
    #     self.submission.to_csv(submission_file, index=False)




import json
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, minkowski, sqeuclidean
from sklearn.metrics import pairwise_distances
import pickle

import nltk

import re
from collections import Counter

import Levenshtein
import distance


def load_test_data():
    attributes_path = './data/test/attributes_test.parquet'
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'
    val_path = './data/test/test.parquet'

    attributes = pd.read_parquet(attributes_path, engine='pyarrow')
    resnet = pd.read_parquet(resnet_path, engine='pyarrow')
    text_and_bert = pd.read_parquet(text_and_bert_path, engine='pyarrow')
    test = pd.read_parquet(val_path, engine='pyarrow')

    return attributes, resnet, text_and_bert, test

# Объекдиняем данные о товарах по парам
def join_attrib_by_pare(pairs_df: pd.DataFrame, attributes_df: pd.DataFrame, text_and_bert_df: pd.DataFrame, resnet_df: pd.DataFrame):
    # Добавляем   аттрибуты
    features_df = (
        pairs_df
        .merge(
            attributes_df
            .add_suffix('1'),
            on="variantid1"
        )
        .merge(
            attributes_df
            .add_suffix('2'),
            on="variantid2"
        )
    )
    # Добавляем текст
    features_df = (
        features_df
        .merge(
            text_and_bert_df
            .add_suffix('1'),
            on="variantid1"
        )
        .merge(
            text_and_bert_df
            .add_suffix('2'),
            on="variantid2"
        )
    )
    # Добавляем картинки
    features_df = (
        features_df
        .merge(
            resnet_df
            .add_suffix('1'),
            on="variantid1"
        )
        .merge(
            resnet_df
            .add_suffix('2'),
            on="variantid2"
        )
    )
    return features_df


#### Сравнение текстового описания
def text_title_iou(name1, name2):
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())
    if len(words1 | words2) == 0:
        iou = 0
    else:
        iou = len(words1 & words2)/ len(words1 | words2)
    return iou


def text_title_iou_filter(name1, name2, skip_rus=False):
    if skip_rus:
        # Допускаются русские буквы только если они идут внутри слова с цифрами или символами иначе убираем их
        pattern = r"[a-zA-Z]+|[\d]+"
    else:
        pattern = r"[a-zA-Z]+|[\d]+|[а-яА-Я]+"
    words1 = set(re.findall(pattern, name1.lower()))
    words2 = set(re.findall(pattern, name2.lower()))

    if len(words1 | words2) == 0:
        iou = 0
    else:
        iou = len(words1 & words2) / len(words1 | words2)
    return iou

#### Подсчитываем IoU 2-грамм и 3-грамм
def calc_iou_ngram(name1, name2, n_gram, skip_rus=False):
    # Убираем все спецсимволы
    #     name1 = re.sub('[^\w\d ]', '', name1.strip())
    #     name2 = re.sub('[^\w\d ]', '', name2.strip())
    pattern = r"[a-zA-Z]+|[\d]+|[а-яА-Я]+"
    name1 = " ".join(sorted(re.findall(pattern, name1.lower())))
    name2 = " ".join(sorted(re.findall(pattern, name2.lower())))
    if skip_rus:
        # Допускаются русские буквы только если они идут внутри слова с цифрами или символами иначе убираем их
        words1 = re.findall(r'(\b[^\s]*[a-zA-Z0-9][^\s]*\b)', name1)
        words2 = re.findall(r'(\b[^\s]*[a-zA-Z0-9][^\s]*\b)', name2)
        name1 = " ".join(words1)
        name2 = " ".join(words2)

    token_name1 = re.findall('\w+', name1.lower())
    token_name2 = re.findall('\w+', name2.lower())

    if len(token_name1) < n_gram or len(token_name2) < n_gram:
        return 0
    bigrams_1 = list(nltk.ngrams(token_name1, n_gram))
    bigrams_2 = list(nltk.ngrams(token_name2, n_gram))
    union = set(bigrams_1) | set(bigrams_2)

    intersection_1 = set(bigrams_1) & set(bigrams_2)
    reverse_bigrams_2 = set([gram[::-1] for gram in bigrams_2]) - set(bigrams_2)
    intersection_2 = set(bigrams_1) & reverse_bigrams_2
    intersection = intersection_1 | intersection_2
    iou_n_gram = len(intersection) / len(union)
    return iou_n_gram


#### Поиск анти-слов, которые говорят, что схожие товары все-таки разные (слова-пристаки Pro Max Plus и прочее)
def calc_anti_words_values(name1, name2):
    words1 = set(re.findall(r'([a-z]+)', name1.lower()))
    words2 = set(re.findall(r'([a-z]+)', name2.lower()))
    if len(words1 | words2) == 0:
        anti_words_val = 0
    else:
        xor_words = words1.symmetric_difference(words2)
        anti_words_val = len(xor_words & top_anti_words) / max(len(words1), len(words2))
    return anti_words_val

#### Дополнительный набор сранвнений текстов
def calc_additional_distance(name1, name2, skip_rus=False):
    # Убираем все спецсимволы
    pattern = r"[a-zA-Z]+|[\d]+|[а-яА-Я]+"
    name1 = " ".join(sorted(re.findall(pattern, name1.lower())))
    name2 = " ".join(sorted(re.findall(pattern, name2.lower())))
    # name1 = re.sub('[^\w\d_ ]', '', name1.lower()).strip()
    # name2 = re.sub('[^\w\d_ ]', '', name2.lower()).strip()
    if skip_rus:
        # Допускаются русские буквы только если они идут внутри слова с цифрами или символами иначе убираем их
        words1 = re.findall( r'(\b[^\s]*[a-zA-Z0-9][^\s]*\b)', name1)
        words2 = re.findall( r'(\b[^\s]*[a-zA-Z0-9][^\s]*\b)', name2)
        name1 = " ".join(words1)
        name2 = " ".join(words2)
    else:
        words1 = re.findall('\w+', name1.lower())
        words2 = re.findall('\w+', name2.lower())
    # Значения дистанций по умолчанию (где-то схожесть, где-то расстояние
    if name1 == "" and name2 == "":
        return 1, 0, 1, 0, -0.08

    lev_ratio = Levenshtein.ratio(name1, name2)
    lev_hamming = Levenshtein.hamming(name1, name2)/max(1, len(name1), len(name2))
    lev_seqratio = Levenshtein.seqratio(words1, words2)
    dist_sorensen = distance.sorensen(name1, name2)
    dist_fast_comp = distance.fast_comp(name1, name2)/max(1, len(name1), len(name2))
    return lev_hamming, lev_ratio, dist_sorensen, lev_seqratio, dist_fast_comp


#### Расчет расстяоний между эмбеддингами изображений
def get_pic_features(main_pic_embeddings_1,
                     main_pic_embeddings_2,
                     percentiles: List[int], metric='euclidean'):
    if main_pic_embeddings_1 is not None and main_pic_embeddings_2 is not None:
        main_pic_embeddings_1 = np.array([x for x in main_pic_embeddings_1])
        main_pic_embeddings_2 = np.array([x for x in main_pic_embeddings_2])

        dist_m = pairwise_distances(
            main_pic_embeddings_1, main_pic_embeddings_2, metric=metric
        )

    else:
        dist_m = np.array([[-1]])

    pair_features = []
    pair_features += np.percentile(dist_m, percentiles).tolist()

    return pair_features


#### Расчет расстяоний между эмбеддингами названий
def text_dense_distances(ozon_embedding, comp_embedding):
    pair_features = []
    if ozon_embedding is None or comp_embedding is None:
        pair_features = [-1, -1]
        return pair_features
    elif len(ozon_embedding) == 0 or len(comp_embedding) == 0:
        pair_features = [-1, -1]
        return pair_features

    ozon_embedding = np.array(ozon_embedding)
    comp_embedding = np.array(comp_embedding)
    if np.sum(ozon_embedding) == 0 and np.sum(comp_embedding) == 0:
        pair_features = [-1, -1]
    elif np.array_equal(ozon_embedding, comp_embedding):
        pair_features = [0, 0]
    else:
        euclidean_value = euclidean(ozon_embedding, comp_embedding)
        cosine_value = cosine(ozon_embedding, comp_embedding)
        pair_features = [euclidean_value, cosine_value]

    return pair_features


#### Переход от сложных 'огненно-красный' в простые ['красный', 'оранжевый']
def calc_color_features(colors_dict:dict, test_features_df:pd.DataFrame ):
    colors_category = sorted(list(set([c for colors in colors_dict.values() for c in colors])))
    colors_columns_name_1 = [f"color1_{i+1}" for i in range(len(colors_category))]
    colors_columns_name_2 = [f"color2_{i+1}" for i in range(len(colors_category))]
    colors_columns_name = [f"color_eq_{i+1}" for i in range(len(colors_category))]

    # Заполняем NaN в цвете товара
    def find_color(name, charact):
        colors = []
        for c in list(colors_dict.keys()):
            if c in name.lower():
                colors.append(c)
        if len(colors) > 0:
            return colors
        return None

    test_features_df["color_parsed1"] = test_features_df.apply(lambda x:
                                                        find_color(x["name1"], x["characteristic_attributes_mapping1"]) if x["color_parsed1"] is None else x["color_parsed1"], axis=1)
    test_features_df["color_parsed2"] = test_features_df.apply(lambda x:
                                                        find_color(x["name2"], x["characteristic_attributes_mapping2"]) if x["color_parsed2"] is None else x["color_parsed2"], axis=1)


    #### Применяем словарь цветов: переход от сложных 'огненно-красный' в простые ['красный', 'оранжевый']
    def get_codes_by_colors(color_parsed):
        color_codes = np.zeros(len(colors_category))
        if color_parsed is None:
            return color_codes
        for src_color in color_parsed:
            if src_color in colors_dict:
                colors = colors_dict[src_color]
            else:
                colors = ['серый']
            for c in colors:
                if c in colors_category:
                    color_codes[colors_category.index(c)] = 1
        return color_codes
    test_features_df[colors_columns_name_1] = test_features_df["color_parsed1"].apply(
        lambda x: pd.Series(get_codes_by_colors(x)))
    test_features_df[colors_columns_name_2] = test_features_df["color_parsed2"].apply(
        lambda x: pd.Series(get_codes_by_colors(x)))

    # Схожесть цветов в товарах
    def color_dist(color_parsed1, color_parsed2):
        color_vec1 = get_codes_by_colors(color_parsed1)
        color_vec2 = get_codes_by_colors(color_parsed2)
        euclidean_value = euclidean(color_vec1, color_vec2)
        cosine_value = cosine(color_vec1, color_vec2)
        binary_compare = np.array(
            [1 if (color_vec1[i] == 1) and (color_vec1[i] == color_vec2[i]) else 0 for i in range(len(color_vec1))])
        intersection = sum(binary_compare)
        union = sum(color_vec2) + sum(color_vec1) - intersection
        if union == 0:
            iou_color = 0
        else:
            iou_color = intersection / union
        return tuple([euclidean_value, (1 - cosine_value), iou_color] + list(binary_compare))


    test_features_df[["color_dist_euclidean", "color_sim_cosine",
                       "iou_color"] + colors_columns_name] = test_features_df.apply(
        lambda x: pd.Series(color_dist(x["color_parsed1"], x["color_parsed2"])), axis=1)
    return test_features_df

#### Расчет Характеристик товаров
def split_attributes(attr_1, attr_2):
    if attr_1 is None or attr_2 is None:
        return (0, 0, 0, 0)
    categories_1 = json.loads(attr_1)
    categories_2 = json.loads(attr_2)

    intersection = 0
    weight_intersection = 0
    # Общее кол-во аттрибутов
    # union = len(set(categories_1.keys()).union(set(categories_2.keys())) )
    union = 0
    for key in categories_1:
        if key in categories_2:
            # union считается как все общие характеристики из расчета что если у товара 2
            # не указана характеристика товра 1, то это не значит что они разные
            union += 1
            if categories_1[key] == categories_2[key]:
                intersection += 1
            weight_intersection += text_title_iou_filter(categories_1[key][0], categories_2[key][0])

    if union == 0:
        iou_attr = 0
        iou_attr_weight = 0
    else:
        iou_attr = intersection / union
        iou_attr_weight = weight_intersection / union
    # IoU для списка категорий, т.е. насколько перечени характеристик
    intersection_dict = len(set(categories_1).intersection(set(categories_2)))
    union_dict = len(set(categories_1).union(set(categories_2)))
    iou_dict = intersection_dict / union_dict

    return iou_attr, iou_attr_weight, iou_dict, iou_attr * iou_dict,


#### Расчет важности характеристики товара на основе DF-IDF
def code_top_df_idf_characteristics(attr_1, attr_2, group_cat, cat_level_3_1, cat_level_3_2):
    top_pop_characts = [i[1] for i in pop_characts_df_idf[group_cat][:TOP_N_characts]]
    vec_attr = np.zeros(TOP_N_characts + 4 + 5)
    if attr_1 is None or attr_2 is None:
        return vec_attr
    attr_1 = json.loads(attr_1)
    attr_2 = json.loads(attr_2)

    if cat_level_3_1 != cat_level_3_2:
        return vec_attr

    for i, cat_name in enumerate(top_pop_characts):
        if cat_name in attr_1 and cat_name in attr_2:
            if attr_1[cat_name] == attr_2[cat_name]:
                vec_attr[i] = 1
            else:
                vec_attr[i] = text_title_iou_filter(" ".join(sorted(attr_1[cat_name])),
                                                    " ".join(sorted(attr_2[cat_name][0])))
    #     print("Расчет биграм и триграм")
    # Расчет биграм и триграм, испоьлзуем все характеристики отосортированные по популярности
    all_pop_characts = [i[1] for i in pop_characts_df_idf[group_cat]]
    char_text_1 = []
    char_text_2 = []
    for i, cat_name in enumerate(all_pop_characts):
        if cat_name in attr_1:
            char_text_1.append(" ".join(sorted(attr_1[cat_name])))
        if cat_name in attr_2:
            char_text_2.append(" ".join(sorted(attr_2[cat_name])))
    char_text_1 = " ".join(sorted(char_text_1))
    char_text_2 = " ".join(sorted(char_text_2))
    gram_2 = calc_iou_ngram(char_text_1, char_text_2, 2)
    gram_3 = calc_iou_ngram(char_text_1, char_text_2, 3)
    gram_4 = calc_iou_ngram(char_text_1, char_text_2, 4)
    gram_5 = calc_iou_ngram(char_text_1, char_text_2, 5)
    vec_attr[TOP_N_characts] = gram_2
    vec_attr[TOP_N_characts + 1] = gram_3
    vec_attr[TOP_N_characts + 2] = gram_4
    vec_attr[TOP_N_characts + 3] = gram_5

    attr_lev_hamming, attr_lev_ratio, attr_dist_sorensen, attr_lev_seqratio, attr_dist_fast_comp = calc_additional_distance(
        char_text_1, char_text_2)

    vec_attr[TOP_N_characts + 4] = attr_lev_hamming
    vec_attr[TOP_N_characts + 5] = attr_lev_ratio
    vec_attr[TOP_N_characts + 6] = attr_dist_sorensen
    vec_attr[TOP_N_characts + 7] = attr_lev_seqratio
    vec_attr[TOP_N_characts + 8] = attr_dist_fast_comp

    return vec_attr


def clean_text(text):
    if pd.isna(text):
        return ''
    # оставляем только слова и цифры
    pattern = r"[a-zA-Z]+|[\d]+|[а-яА-Я]+"
    text = " ".join(re.findall(pattern, text.lower()))
#     # Удаляем HTML-теги
#     text = re.sub(r'<.*?>', '', text)
#     # Заменяем переносы строк и возвраты каретки на пробелы
#     text = re.sub(r'[\r\n]+', ' ', text)
#     # Убираем лишние пробелы
#     text = re.sub(r'\s+', ' ', text).strip()
#     # Удаляем все символы, кроме букв, цифр, пробелов и знаков препинания
#     text = re.sub(r'[^\w\d\s\n.,!?;:(){}\'"-]', '', text)
#     # # Удаляем все символы, кроме букв, цифр и пробелов
#     # text = re.sub(r'[^\w\d ]', '', text)
    return text


# Функция для расчета фич схожести
def calculate_similarity_features(row):
    # Загружаем JSON в словари
    features_1 = json.loads(row['characteristic_attributes_mapping1'])
    features_2 = json.loads(row['characteristic_attributes_mapping2'])

    # Определяем пересечение и разницу ключей
    keys_1 = set(features_1.keys())
    keys_2 = set(features_2.keys())
    common_keys = keys_1 & keys_2
    all_keys = keys_1 | keys_2

    # Фичи схожести
    common_keys_count = len(common_keys)
    prc_equal_keys_count = len(common_keys) / len(all_keys)
    equal_values_count = sum(1 for key in common_keys if features_1[key] == features_2[key])
    if common_keys_count == 0:
        prc_equal_values_count = 0
    else:
        prc_equal_values_count = equal_values_count / common_keys_count

    # Возвращаем фичи как словарь
    return pd.Series({
        'common_keys_count': common_keys_count,
        'prc_equal_keys_count': prc_equal_keys_count,
        'equal_values_count': equal_values_count,
        'prc_equal_values_count': prc_equal_values_count
    })


if __name__ == "__main__":


    # Загружаем тестовые пары и исходные данные
    attributes_df, resnet_df, text_and_bert_df, test_df = load_test_data()

    ### Фичи Артикулы+ISBN
    attributes_df['attributes_dict'] = attributes_df[['variantid', 'characteristic_attributes_mapping']].apply(
        lambda row: {row['variantid']: eval(row['characteristic_attributes_mapping'])}, axis=1)
    attributes_df['characteristic_attributes_mapping_struct'] = attributes_df[
        'characteristic_attributes_mapping'].apply(lambda row: eval(row))
    all_keys = [key for d in attributes_df['characteristic_attributes_mapping_struct'] for key in d.keys()]
    key_counts = Counter(all_keys)
    top_100_keys = [key for key, count in key_counts.most_common(100)]
    # Фичи Артикулы+ISBN
    attributes_df['top_100_keys'] = attributes_df['characteristic_attributes_mapping_struct'].apply(
        lambda x: {k: v for k, v in x.items() if k in top_100_keys})
    attributes_df['attributes_dict'] = attributes_df[['variantid', 'top_100_keys']].apply(
        lambda row: {row['variantid']: row['top_100_keys']}, axis=1)
    result = {}
    for d in attributes_df['attributes_dict']:
        result.update(d)
    df = pd.DataFrame(result)
    df = df.transpose()
    # Функция для преобразования списка в строку или оставления NaN
    def unpack_or_leave(x):
        if isinstance(x, list):  # Если значение - список
            return ', '.join(map(str, x)) if x else np.nan  # Преобразуем в строку, если не пустой, иначе оставляем NaN
        return x  # Возвращаем исходное значение (включая NaN)

    # Применение функции к DataFrame
    df = df.apply(unpack_or_leave)
    df = df.apply(lambda x: x.lower() if isinstance(x, str) else x)[['Артикул', 'ISBN', 'Партномер (артикул производителя)']]
    attributes_df = attributes_df.merge(df, left_on='variantid', right_index=True, how='left')

    # Объединяем тестовые пары с исходными данными
    test_features_df = join_attrib_by_pare(pairs_df=test_df,
                        attributes_df=attributes_df,
                        text_and_bert_df=text_and_bert_df,
                        resnet_df=resnet_df)

    # Фичи Артикулы+ISBN
    # test_features_df['is_equal_article'] = test_features_df['Артикул1'] == test_features_df['Артикул2']
    # test_features_df['is_equal_ISBN'] = test_features_df['ISBN1'] == test_features_df['ISBN2']
    # test_features_df['is_equal_article2'] = test_features_df['Партномер (артикул производителя)1'] == test_features_df['Партномер (артикул производителя)2']

    test_features_df['is_equal_article'] = np.where(test_features_df['Артикул1'] == test_features_df['Артикул2'], 1, 0)
    test_features_df['is_equal_ISBN'] = np.where(test_features_df['ISBN1'] == test_features_df['ISBN2'], 1, 0)
    test_features_df['is_equal_article2'] = np.where(test_features_df['Партномер (артикул производителя)1'] == test_features_df['Партномер (артикул производителя)2'], 1, 0)

    #### Сравнение текстового описания
    test_features_df["iou_names"] = test_features_df.apply(lambda x: text_title_iou(x["name1"], x["name2"]),
                                                                      axis=1)
    test_features_df["iou_names2"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["name1"], x["name2"]), axis=1)
    test_features_df["iou_names2_eng"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["name1"], x["name2"], skip_rus=True), axis=1)

    #### Сравнение описаний
    test_features_df['description1'] = test_features_df['description1'].fillna('None description')
    test_features_df['description2'] = test_features_df['description2'].fillna('None description')
    test_features_df['clear_description1'] = test_features_df['description1'].apply(
        lambda x: clean_text(x)[:1000])
    test_features_df['clear_description2'] = test_features_df['description2'].apply(
        lambda x: clean_text(x)[:1000])

    test_features_df["iou_description"] = test_features_df.apply(
        lambda x: text_title_iou(x["description1"], x["description2"]), axis=1)
    test_features_df["iou_description_2"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["description1"], x["description2"]), axis=1)
    test_features_df["iou_description_2_eng"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["description1"], x["description2"], skip_rus=True), axis=1)

    test_features_df["iou_clear_description"] = test_features_df.apply(
        lambda x: text_title_iou(x["clear_description1"], x["clear_description2"]), axis=1)
    test_features_df["iou_clear_description_2"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["clear_description1"], x["clear_description2"]), axis=1)
    test_features_df["iou_clear_description_2_eng"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["clear_description1"], x["clear_description2"], skip_rus=True), axis=1)

    #### Сравнение характеристик товаров
    # Объединяем характеристики в текст
    def join_characteristic(x):
        return re.sub('[^\w\d:, ]', '', x.strip().replace('\\n', ', '))

    test_features_df['join_characteristic1'] = test_features_df['characteristic_attributes_mapping1'].apply(
        lambda x: join_characteristic(x))
    test_features_df['join_characteristic2'] = test_features_df['characteristic_attributes_mapping2'].apply(
        lambda x: join_characteristic(x))

    test_features_df['clear_join_characteristic1'] = test_features_df['join_characteristic1'].apply(
        lambda x: clean_text(x)[:500])
    test_features_df['clear_join_characteristic2'] = test_features_df['join_characteristic2'].apply(
        lambda x: clean_text(x)[:500])

    test_features_df["iou_join_characteristic"] = test_features_df.apply(
        lambda x: text_title_iou(x["clear_join_characteristic1"], x["clear_join_characteristic2"]), axis=1)
    test_features_df["iou_join_characteristic_2"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["clear_join_characteristic1"], x["clear_join_characteristic2"]), axis=1)
    test_features_df["iou_join_characteristic_2_eng"] = test_features_df.apply(
        lambda x: text_title_iou_filter(x["clear_join_characteristic1"], x["clear_join_characteristic2"],
                                        skip_rus=True), axis=1)

    # расчет схожести характеристик товаров
    test_features_df[['common_keys_count', 'prc_equal_keys_count', 'equal_values_count',
                      'prc_equal_values_count']] = test_features_df.apply(calculate_similarity_features, axis=1)


    #### Подсчитываем IoU 2-грамм и 3-грамм
    test_features_df["iou_1gram_name"] = test_features_df.apply(lambda x: calc_iou_ngram(x["name1"], x["name1"], 1),
                                                                axis=1)
    test_features_df["iou_2gram_name"] = test_features_df.apply(lambda x: calc_iou_ngram(x["name1"], x["name2"], 2),
                                                                axis=1)
    test_features_df["iou_3gram_name"] = test_features_df.apply(lambda x: calc_iou_ngram(x["name1"], x["name2"], 3),
                                                                axis=1)
    test_features_df["iou_4gram_name"] = test_features_df.apply(lambda x: calc_iou_ngram(x["name1"], x["name2"], 4),
                                                                axis=1)
    test_features_df["iou_5gram_name"] = test_features_df.apply(lambda x: calc_iou_ngram(x["name1"], x["name2"], 5),
                                                                axis=1)

    test_features_df["iou_1gram_eng_name"] = test_features_df.apply(
        lambda x: calc_iou_ngram(x["name1"], x["name2"], 1, skip_rus=True), axis=1)
    test_features_df["iou_2gram_eng_name"] = test_features_df.apply(
        lambda x: calc_iou_ngram(x["name1"], x["name2"], 2, skip_rus=True), axis=1)
    test_features_df["iou_3gram_eng_name"] = test_features_df.apply(
        lambda x: calc_iou_ngram(x["name1"], x["name2"], 3, skip_rus=True), axis=1)
    test_features_df["iou_4gram_eng_name"] = test_features_df.apply(
        lambda x: calc_iou_ngram(x["name1"], x["name2"], 4, skip_rus=True), axis=1)
    test_features_df["iou_5gram_eng_name"] = test_features_df.apply(
        lambda x: calc_iou_ngram(x["name1"], x["name2"], 5, skip_rus=True), axis=1)


    #### Загружаем словарь anti_words, рассчитаный при обучении модели
    with open('anti_words.pkl', 'rb') as file:
        anti_words = pickle.load(file)

    N_TOP_ANTIWORD = 100
    top_anti_words = set([w[0] for w in anti_words.most_common(N_TOP_ANTIWORD)])
    #### Поиск анти-слов, которые говорят, что схожие товары все-таки разные (слова-пристаки Pro Max Plus и прочее)
    test_features_df["anti_words_values"] = test_features_df.apply(
        lambda x: calc_anti_words_values(x["name1"], x["name2"]), axis=1)

    #### Дополнительный набор сранвнений текстов
    columns_text_distance = ["lev_hamming", "lev_ratio", "dist_sorensen", "lev_seqratio", "dist_fast_comp", ]
    columns_text_distance_eng = ["lev_hamming_eng", "lev_ratio_eng", "dist_sorensen_eng", "lev_seqratio_eng", "dist_fast_comp_eng", ]
    test_features_df[columns_text_distance] = test_features_df.apply(lambda x: pd.Series(calc_additional_distance(x["name1"], x["name2"])), axis=1 )
    test_features_df[columns_text_distance_eng] = test_features_df.apply(lambda x: pd.Series(calc_additional_distance(x["name1"], x["name2"], skip_rus=True)), axis=1 )

    #### Длина строк
    eps = 1e-6
    test_features_df['diff_len_name'] = abs(test_features_df['name1'].str.len() - test_features_df['name2'].str.len())
    test_features_df['rel_diff_len_name'] = test_features_df['diff_len_name'] / (
                test_features_df['name1'].str.len() + eps)
    test_features_df['diff_len_characteristic'] = abs(
        test_features_df['clear_join_characteristic1'].str.len() - test_features_df[
            'clear_join_characteristic2'].str.len())
    test_features_df['rel_diff_len_characteristic'] = test_features_df['diff_len_characteristic'] / (
                test_features_df['clear_join_characteristic1'].str.len() + eps)
    test_features_df['diff_len_description'] = abs(
        test_features_df['description1'].str.len() - test_features_df['description2'].str.len())
    test_features_df['rel_diff_len_description'] = test_features_df['diff_len_description'] / (
                test_features_df['description1'].str.len() + eps)

    # BM25
    test_features_df['text_attributes1'] = test_features_df['characteristic_attributes_mapping1'].apply(
        lambda x: '\n'.join([f'{k}: {clean_text(", ".join(v))}' for k, v in eval(x).items()]))
    test_features_df['text_attributes2'] = test_features_df['characteristic_attributes_mapping2'].apply(
        lambda x: '\n'.join([f'{k}: {clean_text(", ".join(v))}' for k, v in eval(x).items()]))
    # Применение функции к столбцам и создание нового столбца
    test_features_df['name_description1'] = test_features_df.apply(
        lambda row: re.sub(r'[\r\n\s]+', ' ',
                           f"Наименование: {clean_text(row['name1'])} \n Характеристики: {clean_text(row['text_attributes1'])[:200]} \n Описание: {clean_text(row['description1'])}"),
        axis=1
    )

    test_features_df['name_description2'] = test_features_df.apply(
        lambda row: re.sub(r'[\r\n\s]+', ' ',
                           f"Наименование: {clean_text(row['name2'])} \n Характеристики: {clean_text(row['text_attributes2'])[:200]} \n Описание: {clean_text(row['description2'])}"),
        axis=1
    )
    def compute_bm25_scores(df):
        bm25_scores = []
        for x, y in df.values:
            bm25 = BM25Okapi(clean_text(x).lower().split())
            bm25_scor = bm25.get_scores(clean_text(y).lower().split())
            bm25_scores.append(bm25_scor[0])
        return bm25_scores

    test_features_df['bm25_name'] = compute_bm25_scores(test_features_df[['name1', 'name2']])
    test_features_df['bm25_text_attributes'] = compute_bm25_scores(
        test_features_df[['text_attributes1', 'text_attributes2']])
    ## test_features_df['bm25_name_description'] = compute_bm25_scores(test_features_df[['name_description1','name_description2']])

    # Изображения
    # Для случаев когда у продукта нет дополнительных изображений, то используем основное изображение как дополнительное.
    test_features_df["pic_embeddings_resnet_v11"] = test_features_df.apply(
        lambda x: x["main_pic_embeddings_resnet_v11"] if x["pic_embeddings_resnet_v11"] is None else x[
            "pic_embeddings_resnet_v11"], axis=1)
    test_features_df["pic_embeddings_resnet_v12"] = test_features_df.apply(
        lambda x: x["main_pic_embeddings_resnet_v12"] if x["pic_embeddings_resnet_v12"] is None else x[
            "pic_embeddings_resnet_v12"], axis=1)

    #### Расчет расстяоний между эмбеддингами изображений
    test_features_df["pic_embeddings_resnet_v11"] = test_features_df.apply(lambda x: x["main_pic_embeddings_resnet_v11"] if x["pic_embeddings_resnet_v11"] is None else x["pic_embeddings_resnet_v11"], axis=1)
    test_features_df["pic_embeddings_resnet_v12"] = test_features_df.apply(lambda x: x["main_pic_embeddings_resnet_v12"] if x["pic_embeddings_resnet_v12"] is None else x["pic_embeddings_resnet_v12"], axis=1)

    # Кол-во изображений NEW 01_09_2024
    test_features_df['cnt_img_1'] = test_features_df['pic_embeddings_resnet_v11'].apply(lambda x: len(x))
    test_features_df['cnt_img_2'] = test_features_df['pic_embeddings_resnet_v12'].apply(lambda x: len(x))
    test_features_df['diff_cnt_img'] = abs(test_features_df['cnt_img_1'] - test_features_df['cnt_img_2'])
    test_features_df['rel_diff_cnt_img'] = test_features_df['diff_cnt_img'] / (test_features_df['cnt_img_1'] + eps)

    # Добаляем в доп изображения также главное  NEW 01_09_2024
    test_features_df['pic_embeddings_resnet_v11'] = test_features_df.apply(lambda x: np.concatenate([x['main_pic_embeddings_resnet_v11'], x['pic_embeddings_resnet_v11']]), axis=1)
    test_features_df['pic_embeddings_resnet_v12'] = test_features_df.apply(lambda x: np.concatenate([x['main_pic_embeddings_resnet_v12'], x['pic_embeddings_resnet_v12']]), axis=1)

    get_pic_features_func = partial(
        get_pic_features,
        percentiles=[0, 25, 50, 100],
        metric="euclidean"
    )

    # Сравнение схожести эмбеддингов дополнительных картинок берется несколько перценталей
    for metric in ["cosine"]:
        # print([f"pic_dist_0_perc_{metric}", f"pic_dist_25_perc_{metric}", f"pic_dist_50_perc_{metric}", f"pic_dist_100_perc_{metric}"])
        test_features_df[[f"pic_dist_0_perc_{metric}", f"pic_dist_25_perc_{metric}", f"pic_dist_50_perc_{metric}", f"pic_dist_100_perc_{metric}"]] = (
            test_features_df[["pic_embeddings_resnet_v11", "pic_embeddings_resnet_v12"]].apply(
                lambda x: pd.Series(get_pic_features_func(*x, metric=metric)), axis=1
            )
        )

    # Сравнение схожести эмбеддингов дополнительных картинок берется миниальное расстоние среди пар сравнения
    test_features_df[f"pic_dist_0_perc_sqeuclidean"] = (
        test_features_df[["pic_embeddings_resnet_v11", "pic_embeddings_resnet_v12"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="sqeuclidean")), axis=1
        )
    )


    # Сравнение схожести эмбеддингов главных картинок
    test_features_df[f"main_pic_dist_sqeuclidean"] = (
        test_features_df[["main_pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v12"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="sqeuclidean")), axis=1
        )
    )

    # Сравнение схожести эмбеддингов главных картинок
    test_features_df['main_pic_dist_cityblock'] = (
        test_features_df[["main_pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v12"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="cityblock")), axis=1
        )
    )

    # Сравнение схожести эмбеддингов главных картинок
    test_features_df['main_pic_dist_cosine'] = (
        test_features_df[["main_pic_embeddings_resnet_v11", "main_pic_embeddings_resnet_v12"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="cosine")), axis=1
        )
    )


    #### Расчет расстяоний между эмбеддингами названий
    test_features_df[["euclidean_name_bert_dist", "cosine_name_bert_dist"]] = (
        test_features_df[["name_bert_641", "name_bert_642"]].apply(
            lambda x: pd.Series(text_dense_distances(*x)), axis=1
        )
    )

    #### Кодирование категории цвет
    def color_parsed(x, attr_column_name):
        x_dict = json.loads(x[attr_column_name])
        if 'Цвет товара' in x_dict:
            return x_dict['Цвет товара']
        else:
            return None
    test_features_df['color_parsed1'] = test_features_df.apply(lambda x: color_parsed(x, 'characteristic_attributes_mapping1') , axis=1)
    test_features_df['color_parsed2'] = test_features_df.apply(lambda x: color_parsed(x, 'characteristic_attributes_mapping2') , axis=1)

    #### Словарь цветов: переход от сложных 'огненно-красный' в простые ['красный', 'оранжевый']
    # Загружаем colors_dict, рассчитаный при обучении модели
    with open('colors_dict.pkl', 'rb') as file:
        colors_dict = pickle.load(file)
    #### Переход от сложных 'огненно-красный' в простые ['красный', 'оранжевый']
    test_features_df = calc_color_features(colors_dict, test_features_df)

    ## Категории товаров
    #### Сравнеине  схожести категорий для одинаковых товаров
    def split_cat_product(json_data):
        categories = json.loads(json_data)
        res = [categories[i] for i in categories]
        return res

    test_features_df[["cat_level_1_1", "cat_level_2_1", "cat_level_3_1", "cat_level_4_1"]] = test_features_df[
        "categories1"].apply(lambda x: pd.Series(split_cat_product(x)))
    test_features_df[["cat_level_1_2", "cat_level_2_2", "cat_level_3_2", "cat_level_4_2"]] = test_features_df[
        "categories2"].apply(lambda x: pd.Series(split_cat_product(x)))

    # Признак совпадения категорий
    test_features_df["is_equal_cat_1"] = test_features_df.apply(lambda x: 1 if x["cat_level_1_1"] == x["cat_level_1_2"] else 0, axis=1)
    test_features_df["is_equal_cat_2"] = test_features_df.apply(lambda x: 1 if x["cat_level_2_1"] == x["cat_level_2_2"] else 0, axis=1)
    test_features_df["is_equal_cat_3"] = test_features_df.apply(lambda x: 1 if x["cat_level_3_1"] == x["cat_level_3_2"] else 0, axis=1)
    test_features_df["is_equal_cat_4"] = test_features_df.apply(lambda x: 1 if x["cat_level_4_1"] == x["cat_level_4_2"] else 0, axis=1)

    # Признак совпадения 2х, 3х, 4х категорий
    test_features_df['cat_level_1_2__1'] = test_features_df['cat_level_1_1'] + ' ' + test_features_df['cat_level_2_1']
    test_features_df['cat_level_1_2__2'] = test_features_df['cat_level_1_2'] + ' ' + test_features_df['cat_level_2_2']
    test_features_df['cat_level_1_3__1'] = test_features_df['cat_level_1_1'] + ' ' + test_features_df[
        'cat_level_2_1'] + ' ' + test_features_df['cat_level_3_1']
    test_features_df['cat_level_1_3__2'] = test_features_df['cat_level_1_2'] + ' ' + test_features_df[
        'cat_level_2_2'] + ' ' + test_features_df['cat_level_3_2']
    test_features_df['cat_level_1_4__1'] = test_features_df['cat_level_1_1'] + ' ' + test_features_df[
        'cat_level_2_1'] + ' ' + test_features_df['cat_level_3_1'] + ' ' + test_features_df['cat_level_4_1']
    test_features_df['cat_level_1_4__2'] = test_features_df['cat_level_1_2'] + ' ' + test_features_df[
        'cat_level_2_2'] + ' ' + test_features_df['cat_level_3_2'] + ' ' + test_features_df['cat_level_4_1']

    test_features_df["is_equal_cat_1_2"] = test_features_df.apply(
        lambda x: 1 if x["cat_level_1_2__1"] == x["cat_level_1_2__2"] else 0, axis=1)
    test_features_df["is_equal_cat_1_3"] = test_features_df.apply(
        lambda x: 1 if x["cat_level_1_3__1"] == x["cat_level_1_3__2"] else 0, axis=1)
    test_features_df["is_equal_cat_1_4"] = test_features_df.apply(
        lambda x: 1 if x["cat_level_1_4__1"] == x["cat_level_1_4__2"] else 0, axis=1)

    #### Загружаем типы категорий "fix_cat3" (тут важны индексы категорий такие же которые использовались и при обучении модели)
    # Загружаем fix_cat2
    with open('fix_cat2.pkl', 'rb') as file:
       fix_cat2 = pickle.load(file)
    with open('fix_cat3.pkl', 'rb') as file:
        fix_cat3, fix_cat3_for_model = pickle.load(file)
    with open('fix_cat4.pkl', 'rb') as file:
        fix_cat4 = pickle.load(file)

    test_features_df["cat3_grouped"] = test_features_df["cat_level_3_1"].apply(lambda x: x if x in fix_cat3 else "rest")
    test_features_df["cat3_grouped_id"] = test_features_df["cat3_grouped"].apply(lambda x: fix_cat3.index(x))
    test_features_df["cat3_grouped_model"] = test_features_df["cat_level_3_1"].apply(lambda x: x if x in fix_cat3_for_model else "rest")

    test_features_df["cat4_grouped"] = test_features_df["cat_level_4_1"].apply(lambda x: x if x in fix_cat4 else "rest")
    # CatBoost ругается на русские навзвания в категориях поэтому переводим в индекс
    test_features_df["cat4_grouped_id"] = test_features_df["cat4_grouped"].apply(lambda x: fix_cat4.index(x))

    test_features_df["cat2_grouped"] = test_features_df["cat_level_2_1"].apply(lambda x: x if x in fix_cat2 else "rest")
    # CatBoost ругается на русские навзвания в категориях поэтому переводим в индекс
    test_features_df["cat2_grouped_id"] = test_features_df["cat2_grouped"].apply(lambda x: fix_cat2.index(x))

    test_features_df[["iou_attr", "iou_attr_weight", "iou_dict", "iou_attr_mat_dict"]] = test_features_df[["characteristic_attributes_mapping1","characteristic_attributes_mapping2"]].apply(lambda x: pd.Series(split_attributes(*x)), axis=1)


    #### У каждой категории товаров свои популярные характеристики. Их определили при обучении модели, для инференса просто загружаем популярные характеристики
    # Загружаем pop_characts_df_idf
    with open('pop_characts_df_idf.pkl', 'rb') as file:
        pop_characts_df_idf = pickle.load(file)

    TOP_N_characts = 85
    top_chars_columns = [f"TOP_ch_{i + 1}" for i in range(TOP_N_characts)]
    columns_attr_distance = ["attr_lev_hamming", "attr_lev_ratio", "attr_dist_sorensen", "attr_lev_seqratio",
                         "attr_dist_fast_comp", ]

    test_features_df[top_chars_columns + ["attr_2_gram", "attr_3_gram", "attr_4_gram",
                                       "attr_5_gram"] + columns_attr_distance] = test_features_df.apply(
    lambda x: pd.Series(code_top_df_idf_characteristics(x["characteristic_attributes_mapping1"],
                                                        x["characteristic_attributes_mapping2"],
                                                        x["cat3_grouped_model"], x["cat_level_3_1"],
                                                        x["cat_level_3_2"])), axis=1)


    # Важные характеристики внутри категории первого уровня
    # Загружаем  kv_idf_by_models_c1
    with open('kv_idf_by_models_c1.pkl', 'rb') as file:
        kv_idf_by_models_c1 = pickle.load(file)

    fix_cat1 = list(kv_idf_by_models_c1.keys())
    TOP_N_cats = 100


    def compare_top_characteristics(attr_1, attr_2, cat_level_1_1, cat_level_1_2):
        vec_attr = np.zeros(TOP_N_cats + 2)
        if cat_level_1_1 != cat_level_1_2 or attr_1 is None or attr_2 is None:
            return vec_attr
        attr_1 = json.loads(attr_1)
        attr_2 = json.loads(attr_2)
        count_union_charts = 0
        count_equal_charts = 0
        for i, cat_name in enumerate(kv_idf_by_models_c1[cat_level_1_1][1][:TOP_N_cats]):
            if cat_name in attr_1 and cat_name in attr_2:
                count_union_charts += 1
                if attr_1[cat_name] == attr_2[cat_name]:
                    vec_attr[i] = 1
                    count_equal_charts += 1
                elif len(set(attr_1[cat_name]) & set(attr_2[cat_name])) > 0:
                    vec_attr[i] = 1
                    count_equal_charts += 1
                else:
                    vec_attr[i] = text_title_iou_filter(" ".join(attr_1[cat_name]), " ".join(attr_2[cat_name]))
            elif cat_name in attr_1 or cat_name in attr_2:
                vec_attr[i] = -1
            elif cat_name not in attr_1 and cat_name not in attr_2:
                vec_attr[i] = -2

        if count_union_charts == 0:
            prc_equal_charts = 0
            prc_equal_charts_2 = 0
        else:
            prc_equal_charts = count_equal_charts / count_union_charts
            # только положительные считаем сумму
            prc_equal_charts_2 = count_equal_charts / np.clip(vec_attr, a_min=0, a_max=None).sum()
        vec_attr[TOP_N_cats] = prc_equal_charts
        vec_attr[TOP_N_cats + 1] = prc_equal_charts_2

        return vec_attr


    top_chars_columns = [f"TOP_comp_ch_{i + 1}" for i in range(TOP_N_cats)]
    ext_chars_columns = ["prc_equal_charts", "prc_equal_charts_2"]
    test_features_df[top_chars_columns + ext_chars_columns] = test_features_df.apply(lambda x: pd.Series(
        compare_top_characteristics(x["characteristic_attributes_mapping1"], x["characteristic_attributes_mapping2"],
                                    x["cat_level_1_1"], x["cat_level_1_2"])), axis=1)


    # Важные характеристики внутри категории второго уровня
    # Загружаем  kv_idf_by_models_c2
    with open('kv_idf_by_models_c2.pkl', 'rb') as file:
        kv_idf_by_models_c2 = pickle.load(file)

    fix_cat2 = list(kv_idf_by_models_c2.keys())
    value_TOP_N = 40


    def value_compare_top_characteristics(attr_1, attr_2, cat_level_2_1, cat_level_2_2):
        attr_1 = json.loads(attr_1)
        attr_2 = json.loads(attr_2)

        vec_attr = np.zeros(value_TOP_N)
        vec_attr_lev_hamming = np.ones(value_TOP_N)
        vec_attr_lev_ratio = np.zeros(value_TOP_N)
        vec_attr_dist_sorensen = np.ones(value_TOP_N)
        vec_attr_lev_seqratio = np.zeros(value_TOP_N)
        vec_attr_dist_fast_comp = np.ones(value_TOP_N) * -1

        count_union_charts = 0
        count_equal_charts = 0
        count_1_charts = 0
        count_2_charts = 0
        for i, cat_name in enumerate(kv_idf_by_models_c2[cat_level_2_1][1][:value_TOP_N]):
            if cat_name in attr_1 and cat_name in attr_2:
                count_union_charts += 1
                if attr_1[cat_name] == attr_2[cat_name]:
                    count_equal_charts += 1
                    vec_attr[i] = 1
                    vec_attr_lev_hamming[i] = 0
                    vec_attr_lev_ratio[i] = 1
                    vec_attr_dist_sorensen[i] = 0
                    vec_attr_lev_seqratio[i] = 1
                    vec_attr_dist_fast_comp[i] = 0
                else:

                    text_attr_1 = " ".join(attr_1[cat_name])
                    text_attr_2 = " ".join(attr_2[cat_name])

                    vec_attr[i] = text_title_iou_filter(text_attr_1, text_attr_2)

                    attr_lev_hamming, attr_lev_ratio, attr_dist_sorensen, attr_lev_seqratio, attr_dist_fast_comp = calc_additional_distance(
                        text_attr_1, text_attr_2)
                    vec_attr_lev_hamming[i] = attr_lev_hamming
                    vec_attr_lev_ratio[i] = attr_lev_ratio
                    vec_attr_dist_sorensen[i] = attr_dist_sorensen
                    vec_attr_lev_seqratio[i] = attr_lev_seqratio
                    vec_attr_dist_fast_comp[i] = attr_dist_fast_comp

            elif cat_name in attr_1 or cat_name in attr_2:
                vec_attr[i] = -1
                vec_attr_lev_hamming[i] = -1
                vec_attr_lev_ratio[i] = -1
                vec_attr_dist_sorensen[i] = -1
                vec_attr_lev_seqratio[i] = -1
                vec_attr_dist_fast_comp[i] = -1
                if cat_name in attr_1:
                    count_1_charts += 1
                else:
                    count_2_charts += 1
            elif cat_name not in attr_1 and cat_name not in attr_2:
                vec_attr[i] = -2
                vec_attr_lev_hamming[i] = -2
                vec_attr_lev_ratio[i] = -2
                vec_attr_dist_sorensen[i] = -2
                vec_attr_lev_seqratio[i] = -2
                vec_attr_dist_fast_comp[i] = -2
        if count_1_charts == 0 or count_2_charts == 0:
            prc_equal_by_chart_1 = 0
            prc_equal_by_chart_2 = 0
            prc_equal_by_charts = 0
        else:
            prc_equal_by_chart_1 = count_equal_charts / count_1_charts
            prc_equal_by_chart_2 = count_equal_charts / count_2_charts
            prc_equal_by_charts = count_equal_charts / (count_1_charts + count_2_charts)

        if count_union_charts == 0:
            prc_equal_charts = 0
            prc_equal_charts_2 = 0
        else:
            prc_equal_charts = count_equal_charts / count_union_charts
            # только положительные считаем сумму
            prc_equal_charts_2 = count_equal_charts / np.clip(vec_attr, a_min=0, a_max=None).sum()
        add_statistic = np.array(
            (prc_equal_by_chart_1, prc_equal_by_chart_2, prc_equal_by_charts, prc_equal_charts, prc_equal_charts_2,))
        return np.concatenate((vec_attr, vec_attr_lev_hamming, vec_attr_lev_ratio, vec_attr_dist_sorensen,
                               vec_attr_lev_seqratio, vec_attr_dist_fast_comp, add_statistic))


    top_chars_columns_cat_level_2 = [f"TOP_cat_2_ch_vec_attr_{i + 1}" for i in range(value_TOP_N)] + [
        f"TOP_cat_2_ch_vec_attr_lev_hamming_{i + 1}" for i in range(value_TOP_N)] + [
                                        f"TOP_cat_2_ch_vec_attr_lev_ratio_{i + 1}" for i in range(value_TOP_N)] + [
                                        f"TOP_cat_2_ch_vec_attr_dist_sorensen_{i + 1}" for i in range(value_TOP_N)] + [
                                        f"TOP_cat_2_ch_vec_attr_lev_seqratio_{i + 1}" for i in range(value_TOP_N)] + [
                                        f"TOP_cat_2_ch_vec_attr_dist_fast_comp_{i + 1}" for i in range(value_TOP_N)]
    ext_chars_columns_cat_level_2 = ['prc_equal_by_chart_1_cat_level_2', 'prc_equal_by_chart_2_cat_level_2',
                                     'prc_equal_by_charts_cat_level_2', 'prc_equal_charts_cat_level_2',
                                     'prc_equal_charts_2_cat_level_2', ]

    test_features_df[top_chars_columns_cat_level_2 + ext_chars_columns_cat_level_2] = test_features_df.apply(
        lambda x: pd.Series(value_compare_top_characteristics(x["characteristic_attributes_mapping1"],
                                                              x["characteristic_attributes_mapping2"],
                                                              x["cat_level_2_1"], x["cat_level_2_2"])), axis=1)

    # Доп фичи на базе характеристик для 1го уровня
    value_TOP_N_cat_1 = 40
    def value_compare_top_characteristics_cat_1(attr_1, attr_2, cat_level_1_1, cat_level_1_2):
        attr_1 = json.loads(attr_1)
        attr_2 = json.loads(attr_2)

        vec_attr = np.zeros(value_TOP_N_cat_1)
        vec_attr_lev_hamming = np.ones(value_TOP_N_cat_1)
        vec_attr_lev_ratio = np.zeros(value_TOP_N_cat_1)
        vec_attr_dist_sorensen = np.ones(value_TOP_N_cat_1)
        vec_attr_lev_seqratio = np.zeros(value_TOP_N_cat_1)
        vec_attr_dist_fast_comp = np.ones(value_TOP_N_cat_1) * -1

        count_union_charts = 0
        count_equal_charts = 0
        count_1_charts = 0
        count_2_charts = 0
        for i, cat_name in enumerate(kv_idf_by_models_c1[cat_level_1_1][1][:value_TOP_N_cat_1]):
            if cat_name in attr_1 and cat_name in attr_2:
                count_union_charts += 1
                if attr_1[cat_name] == attr_2[cat_name]:
                    count_equal_charts += 1
                    vec_attr[i] = 1
                    vec_attr_lev_hamming[i] = 0
                    vec_attr_lev_ratio[i] = 1
                    vec_attr_dist_sorensen[i] = 0
                    vec_attr_lev_seqratio[i] = 1
                    vec_attr_dist_fast_comp[i] = 0
                else:

                    text_attr_1 = " ".join(attr_1[cat_name])
                    text_attr_2 = " ".join(attr_2[cat_name])

                    vec_attr[i] = text_title_iou_filter(text_attr_1, text_attr_2)

                    attr_lev_hamming, attr_lev_ratio, attr_dist_sorensen, attr_lev_seqratio, attr_dist_fast_comp = calc_additional_distance(
                        text_attr_1, text_attr_2)
                    vec_attr_lev_hamming[i] = attr_lev_hamming
                    vec_attr_lev_ratio[i] = attr_lev_ratio
                    vec_attr_dist_sorensen[i] = attr_dist_sorensen
                    vec_attr_lev_seqratio[i] = attr_lev_seqratio
                    vec_attr_dist_fast_comp[i] = attr_dist_fast_comp

            elif cat_name in attr_1 or cat_name in attr_2:
                vec_attr[i] = -1
                vec_attr_lev_hamming[i] = -1
                vec_attr_lev_ratio[i] = -1
                vec_attr_dist_sorensen[i] = -1
                vec_attr_lev_seqratio[i] = -1
                vec_attr_dist_fast_comp[i] = -1
                if cat_name in attr_1:
                    count_1_charts += 1
                else:
                    count_2_charts += 1
            elif cat_name not in attr_1 and cat_name not in attr_2:
                vec_attr[i] = -2
                vec_attr_lev_hamming[i] = -2
                vec_attr_lev_ratio[i] = -2
                vec_attr_dist_sorensen[i] = -2
                vec_attr_lev_seqratio[i] = -2
                vec_attr_dist_fast_comp[i] = -2
        if count_1_charts == 0 or count_2_charts == 0:
            prc_equal_by_chart_1 = 0
            prc_equal_by_chart_2 = 0
            prc_equal_by_charts = 0
        else:
            prc_equal_by_chart_1 = count_equal_charts / count_1_charts
            prc_equal_by_chart_2 = count_equal_charts / count_2_charts
            prc_equal_by_charts = count_equal_charts / (count_1_charts + count_2_charts)

        if count_union_charts == 0:
            prc_equal_charts = 0
            prc_equal_charts_2 = 0
        else:
            prc_equal_charts = count_equal_charts / count_union_charts
            # только положительные считаем сумму
            prc_equal_charts_2 = count_equal_charts / np.clip(vec_attr, a_min=0, a_max=None).sum()
        add_statistic = np.array(
            (prc_equal_by_chart_1, prc_equal_by_chart_2, prc_equal_by_charts, prc_equal_charts, prc_equal_charts_2,))
        return np.concatenate((vec_attr, vec_attr_lev_hamming, vec_attr_lev_ratio, vec_attr_dist_sorensen,
                               vec_attr_lev_seqratio, vec_attr_dist_fast_comp, add_statistic))


    top_chars_columns_cat_level_1 = [f"TOP_cat_1_ch_vec_attr_{i + 1}" for i in range(value_TOP_N_cat_1)] + [
        f"TOP_cat_1_ch_vec_attr_lev_hamming_{i + 1}" for i in range(value_TOP_N_cat_1)] + [
                                        f"TOP_cat_1_ch_vec_attr_lev_ratio_{i + 1}" for i in
                                        range(value_TOP_N_cat_1)] + [
                                        f"TOP_cat_1_ch_vec_attr_dist_sorensen_{i + 1}" for i in
                                        range(value_TOP_N_cat_1)] + [
                                        f"TOP_cat_1_ch_vec_attr_lev_seqratio_{i + 1}" for i in
                                        range(value_TOP_N_cat_1)] + [
                                        f"TOP_cat_1_ch_vec_attr_dist_fast_comp_{i + 1}" for i in
                                        range(value_TOP_N_cat_1)]
    ext_chars_columns_cat_level_1 = ['prc_equal_by_chart_1_cat_level_1', 'prc_equal_by_chart_2_cat_level_1',
                                     'prc_equal_by_charts_cat_level_1', 'prc_equal_charts_cat_level_1',
                                     'prc_equal_charts_2_cat_level_1', ]

    test_features_df[top_chars_columns_cat_level_1 + ext_chars_columns_cat_level_1] = test_features_df.apply(
        lambda x: pd.Series(value_compare_top_characteristics_cat_1(x["characteristic_attributes_mapping1"],
                                                                    x["characteristic_attributes_mapping2"],
                                                                    x["cat_level_1_1"], x["cat_level_1_2"])), axis=1)

    #### Доп фичи по картинкам
    test_features_df['mean_pic_embeddings_resnet_v1'] = test_features_df['pic_embeddings_resnet_v11'].fillna(
        np.nan).apply(lambda x: [np.mean(x, axis=0)] if type(x) == np.ndarray else x)
    test_features_df['mean_pic_embeddings_resnet_v2'] = test_features_df['pic_embeddings_resnet_v12'].fillna(
        np.nan).apply(lambda x: [np.mean(x, axis=0)] if type(x) == np.ndarray else x)
    # Сравнение схожести средних эмбеддингов доп картинок sqeuclidean
    test_features_df[f"mean_add_pic_dist_sqeuclidean"] = (
        test_features_df[["mean_pic_embeddings_resnet_v1", "mean_pic_embeddings_resnet_v2"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="sqeuclidean")), axis=1
        )
    )

    # Сравнение схожести средних эмбеддингов доп картинок cityblock
    test_features_df['mean_add_pic_dist_cityblock'] = (
        test_features_df[["mean_pic_embeddings_resnet_v1", "mean_pic_embeddings_resnet_v2"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="cityblock")), axis=1
        )
    )
    # Сравнение схожести средних эмбеддингов доп картинок cosine
    test_features_df['mean_add_dist_cosine'] = (
        test_features_df[["mean_pic_embeddings_resnet_v1", "mean_pic_embeddings_resnet_v2"]].apply(
            lambda x: pd.Series(get_pic_features_func(*x, percentiles=[0], metric="cosine")), axis=1
        )
    )

    #### Доп фичи по тексту
    # Общая длина префикса
    def common_prefix_length(str1, str2):
        pattern = r"[a-zA-Z]+|[\d]+|[а-яА-Я]+"
        str1 = " ".join(sorted(re.findall(pattern, str1.lower())))
        str2 = " ".join(sorted(re.findall(pattern, str2.lower())))
        length = min(len(str1), len(str2))
        for i in range(length):
            if str1[i] != str2[i]:
                return i, i / length
        return length, 1

    test_features_df['clear_name1'] = test_features_df['name1'].apply(lambda x: clean_text(x))
    test_features_df['clear_name2'] = test_features_df['name2'].apply(lambda x: clean_text(x))
    test_features_df['clear_description1'] = test_features_df['description1'].apply(lambda x: clean_text(x))
    test_features_df['clear_description2'] = test_features_df['description2'].apply(lambda x: clean_text(x))

    test_features_df['is_equal_description'] = test_features_df['clear_description1'] == test_features_df[
        'clear_description2']
    test_features_df['is_equal_name'] = test_features_df['clear_name1'] == test_features_df['clear_name2']
    test_features_df['is_equal_characteristic'] = test_features_df['clear_join_characteristic1'] == test_features_df[
        'clear_join_characteristic2']

    test_features_df[['prefix_length_name', 'prc_prefix_length_name']] = test_features_df.apply(
        lambda x: pd.Series(common_prefix_length(x['name1'], x['name2'])), axis=1)

    test_features_df[['prefix_length_description', 'prc_prefix_length_description']] = test_features_df.apply(
        lambda x: pd.Series(common_prefix_length(x['clear_description1'], x['clear_description2'])), axis=1)

    test_features_df[['prefix_length_character', 'prc_prefix_length_character']] = test_features_df.apply(
        lambda x: pd.Series(common_prefix_length(x['clear_join_characteristic1'], x['clear_join_characteristic2'])),
        axis=1)

    # Добавляем фичи по категориям
    # Загружаем dict_cats_encoding
    with open('dict_cats_encoding.pkl', 'rb') as file:
        dict_cats_encoding = pickle.load(file)
    # КОДИРУЕМ категории
    test_features_df['code_categories_1'] = test_features_df['cat_level_1_1'].apply(
        lambda x: dict_cats_encoding[0][x] if x in dict_cats_encoding[0] else -1)
    test_features_df['code_categories_2'] = test_features_df['cat_level_2_1'].apply(
        lambda x: dict_cats_encoding[1][x] if x in dict_cats_encoding[1] else -1)
    test_features_df['code_categories_3'] = test_features_df['cat_level_3_1'].apply(
        lambda x: dict_cats_encoding[2][x] if x in dict_cats_encoding[2] else -1)
    test_features_df['code_categories_4'] = test_features_df['cat_level_4_1'].apply(
        lambda x: dict_cats_encoding[3][x] if x in dict_cats_encoding[3] else -1)

    # # Добавляем фичи  по  берт
    # Функция для удаления HTML-тегов и очистки текста
    # def clean_text_bert(text):
    #     if pd.isna(text):
    #         return ''
    #     text = re.sub(r'<.*?>', '', text)  # Удаляем HTML-теги
    #     text = re.sub(r'[\r\n]+', ' ', text)  # Удаляем переносы строк и возвраты каретки
    #     text = re.sub(r'\s+', ' ', text).strip()  # Убираем лишние пробелы
    #     text = re.sub(r'[^\w\d\s\n.,!?;:(){}\'"-]', '', text)  # Удаляем все символы, кроме букв, цифр и пробелов
    #     return text.lower()
    #
    #
    # # Функция для вычисления косинусного сходства с помощью модели Sentence-BERT
    # def cosine_bert(x, y, model, device, batch_size=512):
    #     all_similarities = []
    #     for i in tqdm(range(0, len(x), batch_size), desc="Encoding Batches"):
    #         batch_texts1 = x[i:i + batch_size]
    #         batch_texts2 = y[i:i + batch_size]
    #         with torch.no_grad():
    #             embeddings1 = model.encode(batch_texts1, convert_to_tensor=True, device=device).cpu()
    #             embeddings2 = model.encode(batch_texts2, convert_to_tensor=True, device=device).cpu()
    #             chunk_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    #         all_similarities.append(chunk_similarities.diagonal())
    #         torch.cuda.empty_cache()
    #     return torch.cat(all_similarities)
    #
    #
    # # Основная функция, которую можно импортировать
    # def process_data(attributes_df, text_and_bert_df, union_features_df, model_path):
    #     # Очистка текста
    #     attributes_df['text_attributes_'] = attributes_df['characteristic_attributes_mapping'].progress_apply(
    #         lambda x: '\n'.join([f'{k}: {clean_text_bert(", ".join(v))}' for k, v in eval(x).items()])
    #     )
    #
    #     df_text = text_and_bert_df.merge(attributes_df, on='variantid')
    #     df_text = df_text.rename(columns={'name': 'name_'})
    #     df_text['name_description_'] = df_text.progress_apply(
    #         lambda row: re.sub(r'[\r\n\s]+', ' ',
    #                            f"Наименование: {clean_text_bert(row['name_'])}\nХарактеристики: {clean_text_bert(row['text_attributes_'])[:200]}\nОписание: {clean_text_bert(row['description'])}"),
    #         axis=1
    #     )
    #
    #     # Подгружаем модель
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = SentenceTransformer(model_path, device=device)
    #
    #     # Объединение данных
    #     union_features_df = union_features_df.merge(
    #         df_text[['variantid', 'name_description_', 'text_attributes_', 'name_']].add_suffix('1'), on='variantid1')
    #     union_features_df = union_features_df.merge(
    #         df_text[['variantid', 'name_description_', 'text_attributes_', 'name_']].add_suffix('2'), on='variantid2')
    #
    #     # Получаем списки текстов
    #     batch_descriptions1 = union_features_df["name_description_1"].tolist()
    #     batch_descriptions2 = union_features_df["name_description_2"].tolist()
    #     batch_names1 = union_features_df["name_1"].progress_apply(lambda row: clean_text_bert(row)).tolist()
    #     batch_names2 = union_features_df["name_2"].progress_apply(lambda row: clean_text_bert(row)).tolist()
    #     batch_attributes1 = union_features_df["text_attributes_1"].progress_apply(
    #         lambda row: clean_text_bert(row)).tolist()
    #     batch_attributes2 = union_features_df["text_attributes_2"].progress_apply(
    #         lambda row: clean_text_bert(row)).tolist()
    #
    #     # Вычисляем косинусное сходство
    #     cosine_name_similarities = cosine_bert(batch_names1, batch_names2, model, device)
    #     cosine_attribute_similarities = cosine_bert(batch_attributes1, batch_attributes2, model, device)
    #     cosine_description_similarities = cosine_bert(batch_descriptions1, batch_descriptions2, model, device)
    #
    #     # Добавляем результаты в датафрейм
    #     union_features_df["cosine_bert_description_text"] = cosine_description_similarities.tolist()
    #     union_features_df["cosine_bert_name_text"] = cosine_name_similarities.tolist()
    #     union_features_df["cosine_bert_attribute_text"] = cosine_attribute_similarities.tolist()
    #     union_features_df = union_features_df.drop(['name_description_1', 'text_attributes_1', 'name_description_2',
    #                                                 'text_attributes_2', 'name_1', 'name_2'], axis=1)
    #     return union_features_df
    #
    # # Логика для запуска скрипта напрямую
    # model_path = './Bert_v3'
    # #запуск
    # test_features_df = process_data(
    #     attributes, # заменить своим датафрейм с атрибутами
    #     text_and_bert, #датафрейм с атрибутами
    #     test_features_df, #датафрейм с фичиами
    #     model_path #путь к папке с моделью
    # )


    # cross_model_filename = 'cross_val_models_07_09_2024__vb8_3_4.pkl'
    model_filename_cat_0 = 'cat_zero__20_models_08_09_2024__vb9_1_3.pkl'
    model_filename_cat_1 = 'cat_1__3_models_08_09_2024__vb9_1_3.pkl'
    model_filename_cat_2 = 'cat_2__20_models_08_09_2024__vb9_1_3.pkl'
    # tar = tarfile.open(f"{cross_model_filename}.tar.gz", "r:gz")
    # tar.extractall()
    # tar.close()
    # tar = tarfile.open(f"{model_filename}.tar.gz", "r:gz")
    # tar.extractall()
    # tar.close()

    # Добавляем фичу   от моделей кроссвалидации
    # загружаем модели кроссвалидации
    # Сохраняем  модель
    # with open(cross_model_filename, 'rb') as file:
    #     models_by_cat_crossval = pickle.load(file)
    # cv_features_columns = models_by_cat_crossval[list(models_by_cat_crossval.keys())[0]][0].feature_names_
    # def predict_by_cross_val_models(test_df:pd.DataFrame, cv_features_columns):
    #     for cat_name in models_by_cat_crossval:
    #         for i, model in enumerate(models_by_cat_crossval[cat_name]):
    #             test_df.loc[test_df['cat_level_1_1'] == cat_name, f'{i}_cv_predict'] = model.predict_proba(
    #                 test_df[test_df['cat_level_1_1'] == cat_name][cv_features_columns])[:, 1]
    #     cv_predict_columns = ['0_cv_predict', '1_cv_predict', '2_cv_predict', '3_cv_predict', '4_cv_predict']
    #     test_df['cv_predict'] = test_df[cv_predict_columns].mean(axis=1)
    #     test_df = test_df.drop(columns=cv_predict_columns, errors='ignore')
    #     return test_df
    # test_features_df = predict_by_cross_val_models(test_features_df, cv_features_columns)

    with open(model_filename_cat_0, 'rb') as file:
        zero_model = pickle.load(file)
    zero_features_columns = zero_model.feature_names_
    test_features_df['target_cat_0'] = zero_model.predict_proba(test_features_df[zero_features_columns])[:, 1]


    # submission = Submission(test_features_path='test_df.parquet')
    # submission = Submission(test_features_path='app/test_features_df.parquet')
    submission_cat_1 = Submission(test_features_df=test_features_df, cat_level=1)
    submission_cat_1.load_model(model_path=model_filename_cat_1)
    test_features_df = submission_cat_1.predict()


    submission_cat_2 = Submission(test_features_df=test_features_df, cat_level=2)
    submission_cat_2.load_model(model_path=model_filename_cat_2)
    test_features_df = submission_cat_2.predict()
    # Считаем среднее предсказание по моделям
    # Даем вес моделям
    # test_features_df['target'] = np.mean((
    #     test_features_df['target_cat_0'].values * 0.7,
    #     test_features_df['target_cat_1'].values * 2.1 ,
    #     test_features_df['target_cat_2'].values * 3), axis=0)
    # Даем вес моделям
    test_features_df['target'] = np.average((
        test_features_df['target_cat_0'].values,
        test_features_df['target_cat_1'].values,
        test_features_df['target_cat_2'].values), axis=0, weights=[0.7, 2.1, 3])

    submission_df = test_features_df[['variantid1', 'variantid2', 'target']]
    submission_df.to_csv('data/submission.csv', index=False)
