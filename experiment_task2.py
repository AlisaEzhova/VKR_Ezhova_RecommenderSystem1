"""
experiment_task2.py
Задача 2: натравить систему на датасет Yandex Yambda-50M
"""

from system import RecommenderSystem
from datasets import load_dataset
from collections import defaultdict
import random

print("Задача 2: Загрузка датасета и натравливание системы")

# Шаг 1: загрузка датасета
print("\n1. Загрузка датасета Yandex Yambda-50M")
# Загружаем только файл с лайками (он маленький, ~7 MB)
likes_ds = load_dataset(
    "yandex/yambda", 
    data_dir="flat/50m", 
    data_files="likes.parquet",
    split="train"
)

# Шаг 2: формирование структуры user_likes
# user_likes = {id пользователя: множество id треков, которые он лайкнул}
print("Формирование данных о лайках пользователей")
user_likes = defaultdict(set)
for example in likes_ds:
    user_likes[example['uid']].add(example['item_id'])

print(f"Всего пользователей в датасете: {len(user_likes)}")

# Шаг 3: выборка 1000 пользователей (для экономии памяти и времени)
# Берём первых 1000 пользователей из списка
print("\n2. Выборка 1000 пользователей")
all_users = list(user_likes.keys())
selected_users = all_users[:1000]
print(f"Выбрано пользователей: {len(selected_users)}")

# Фильтруем данные: оставляем только выбранных пользователей
filtered_user_likes = {uid: user_likes[uid] for uid in selected_users if uid in user_likes}

# Шаг 4: настройка конфигурации системы
# Все параметры можно менять. max_neighbors = None означает "все соседи"
print("\n3. Настройка системы")
config = {
    'max_recommendations': 10,    # рекомендовать 10 треков
    'fail_limit': 3,              # лимит отказов: 3 неудачные рекомендации
    'strategy': 'greedy',         # жадная сортировка по весу
    'min_common': 1,              # достаточно 1 общего трека для сходства
    'max_neighbors': None,        # не ограничиваем количество соседей
    
    # Гипотезы: H1, H2, H4 включены (базовые)
    # H3, H5, H6 выключены (требуют дополнительных данных)
    'use_H1': True,
    'use_H2': True,
    'use_H3': False,
    'use_H4': True,
    'use_H5': False,
    'use_H6': False,
}

# Шаг 5: создание и обучение системы
system = RecommenderSystem(config)
system.fit(filtered_user_likes)

print(f"Конфигурация: max_recommendations={config['max_recommendations']}, "
      f"fail_limit={config['fail_limit']}, strategy={config['strategy']}, "
      f"max_neighbors={config['max_neighbors']}")

# Шаг 6: получение рекомендаций для нескольких пользователей
print("\n4. Рекомендации для пользователей из выборки")

# Выбираем 5 случайных пользователей для демонстрации
test_users = random.sample(selected_users, min(5, len(selected_users)))

for user_id in test_users:
    recommendations, _ = system.recommend(target_user=user_id)
    
    # Выводим информацию о пользователе и его рекомендациях
    user_likes_count = len(filtered_user_likes.get(user_id, set()))
    rec_count = len(recommendations)
    
    print(f"\nПользователь {user_id}:")
    print(f"  Лайкнул треков: {user_likes_count}")
    print(f"  Получено рекомендаций: {rec_count}")
    
    if rec_count > 0:
        # Показываем первые 5 рекомендаций (чтобы не загромождать вывод)
        print(f"  Первые 5 рекомендаций: {recommendations[:5]}")
    else:
        print("  Рекомендаций нет (нет подходящих соседей)")

# Шаг 7: итоговая статистика
print("\n5. Статистика по системе")
total_users_with_recommendations = 0
total_recommendations = 0

for user_id in selected_users[:100]:  # Проверяем первых 100 пользователей
    recommendations, _ = system.recommend(target_user=user_id)
    if recommendations:
        total_users_with_recommendations += 1
        total_recommendations += len(recommendations)

print(f"Из 100 проверенных пользователей:")
print(f"  - Получили рекомендации: {total_users_with_recommendations}")
print(f"  - Всего рекомендаций: {total_recommendations}")
print(f"  - В среднем на пользователя: {total_recommendations / 100:.2f}")

print("\nЗадача 2 выполнена. Система натренирована на датасете.")