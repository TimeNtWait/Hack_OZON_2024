## E-CUP Ozon Tech 2024 (4 место из 110 команд)
### Кейс: Матчинг товаров 
##### Кейсодержатель: ОЗОН
https://codenrock.com/contests/e-cup-ozontech-challenge#/leaderboards/16

***
### Презентация проекта
[Итоговая презентация решения на защите](https://docs.google.com/presentation/d/1suqygadTEOicOtecfsRRwUDXJOZlaZIn/edit?usp=sharing&ouid=113491937784577068477&rtpof=true&sd=true)


### Структура репозитария:
* fix_cat2.pkl - словарь категорий 2го уровня  
* fix_cat3.pkl - словарь категорий 3го уровня  
* fix_cat4.pkl - словарь категорий 4го уровня  
* pop_characts_df_idf.pkl - популярные характеристики
* colors_dict.pkl - словарь цветов
* requirements.txt - зависимости
* kv_idf_by_models_c1.pkl - популярные характеристики внутри категорий 1го уровня
* kv_idf_by_models_c2.pkl - популярные характеристики внутри категорий 2го уровня
* dict_cats_encoding.pkl - енкодер категорий
* anti_words.pkl - словарь "антислов"
* make_submission.py - сборка submission.csv на базе тестовых данных

### Модели
* cat_zero__20_models_08_09_2024__vb9_1_3.pkl
* cat_1__3_models_08_09_2024__vb9_1_3.pkl
* cat_2__20_models_08_09_2024__vb9_1_3.pkl

### Код обучения моделей
* train_model_v9_3.ipynb

