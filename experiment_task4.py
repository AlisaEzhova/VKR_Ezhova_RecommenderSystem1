"""
experiment_task4.py

ЗАДАЧА 4
Сравнение различных параметров системы
Анализ того, как меняются рекомендации при изменении параметров
"""

from system import RecommenderSystem
from datasets import load_dataset
from collections import defaultdict
import random

print("ЗАДАЧА 4: СРАВНЕНИЕ РАЗЛИЧНЫХ ПАРАМЕТРОВ СИСТЕМЫ")

# ЧАСТЬ 1: ЗАГРУЗКА ДАННЫХ (как в задаче 2)

print("\n[1/3] Загрузка данных...")

likes_ds = load_dataset(
    "yandex/yambda", 
    data_dir="flat/50m", 
    data_files="likes.parquet",
    split="train"
)

print("Формирование данных о лайках пользователей...")
user_likes = defaultdict(set)
for example in likes_ds:
    user_likes[example['uid']].add(example['item_id'])

all_users = list(user_likes.keys())
selected_users = all_users[:500]  # 500 пользователей
filtered_user_likes = {uid: user_likes[uid] for uid in selected_users}

print(f"  Всего пользователей: {len(user_likes)}")
print(f"  Выбрано пользователей: {len(filtered_user_likes)}")

# Выбираем одного пользователя для демонстрации
demo_user = random.choice(selected_users)
print(f"\n  Демонстрационный пользователь: {demo_user}")
print(f"  Его лайки (первые 10): {list(filtered_user_likes[demo_user])[:10]}")

# ЧАСТЬ 2: СРАВНЕНИЕ РАЗНЫХ ПАРАМЕТРОВ

print("\n[2/3] Сравнение конфигураций...")

# Список конфигураций для сравнения
configs = [
    # 1. Разные fail_limit (сколько раз показывать трек, если не лайкнул)
    {'name': ' fail_limit = 1 (жесткий лимит)', 'fail_limit': 1, 'strategy': 'greedy', 'max_neighbors': None},
    {'name': ' fail_limit = 3 (базовый)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None},
    {'name': ' fail_limit = 5', 'fail_limit': 5, 'strategy': 'greedy', 'max_neighbors': None},
    {'name': ' fail_limit = 10', 'fail_limit': 10, 'strategy': 'greedy', 'max_neighbors': None},
    {'name': ' fail_limit = ∞ (без лимита)', 'fail_limit': 999999, 'strategy': 'greedy', 'max_neighbors': None},
    
    # 2. Разные стратегии
    {'name': ' Стратегия: greedy (жадная)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None},
    {'name': ' Стратегия: random (случайная)', 'fail_limit': 3, 'strategy': 'random', 'max_neighbors': None},
    
    # 3. Ограничение количества соседей
    {'name': ' max_neighbors = 2', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': 2},
    {'name': ' max_neighbors = 5', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': 5},
    {'name': ' max_neighbors = None (все)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None},
    
    # 4. Разные гипотезы
    {'name': ' Базовый (H1+H2+H4)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None, 'use_H3': False, 'use_H6': False},
    {'name': ' С H3 (сильные прослушивания)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None, 'use_H3': True, 'use_H6': False},
    {'name': ' С H6 (асимметрия)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None, 'use_H3': False, 'use_H6': True},
    {'name': ' Все гипотезы (H1-H6)', 'fail_limit': 3, 'strategy': 'greedy', 'max_neighbors': None, 'use_H3': True, 'use_H6': True},
]

# Сохраняем результаты для демонстрационного пользователя
demo_results = []

for cfg in configs:
    # Настраиваем конфигурацию
    config = {
        'max_recommendations': 10,
        'fail_limit': cfg['fail_limit'],
        'strategy': cfg['strategy'],
        'min_common': 1,
        'max_neighbors': cfg['max_neighbors'],
        'use_H1': True,
        'use_H2': True,
        'use_H3': cfg.get('use_H3', False),
        'use_H4': True,
        'use_H5': False,
        'use_H6': cfg.get('use_H6', False),
    }
    
    # Создаём и обучаем систему
    system = RecommenderSystem(config)
    system.fit(filtered_user_likes)
    
    # Получаем рекомендации для демонстрационного пользователя
    recommendations, _ = system.recommend(target_user=demo_user)
    
    demo_results.append({
        'name': cfg['name'],
        'recommendations': recommendations[:5],  # только первые 5
        'count': len(recommendations)
    })

# ЧАСТЬ 3: ВЫВОД РЕЗУЛЬТАТОВ

print("\n[3/3] Результаты сравнения для демонстрационного пользователя")
print(f"  Пользователь: {demo_user}")
print(f"  Его лайки (первые 10): {list(filtered_user_likes[demo_user])[:10]}")
print("КАК МЕНЯЮТСЯ РЕКОМЕНДАЦИИ ПРИ ИЗМЕНЕНИИ ПАРАМЕТРОВ")

for res in demo_results:
    print(f"\n{res['name']}:")
    print(f"  Получено рекомендаций: {res['count']}")
    if res['count'] > 0:
        print(f"  Первые 5 рекомендаций: {res['recommendations']}")
    else:
        print("  Рекомендаций нет")

# ЧАСТЬ 4: ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА

print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА ПО ВСЕМ ПОЛЬЗОВАТЕЛЯМ")

# Проверяем базовую конфигурацию на всех пользователях
base_config = {
    'max_recommendations': 10,
    'fail_limit': 3,
    'strategy': 'greedy',
    'min_common': 1,
    'max_neighbors': None,
    'use_H1': True, 'use_H2': True, 'use_H3': False,
    'use_H4': True, 'use_H5': False, 'use_H6': False,
}

system = RecommenderSystem(base_config)
system.fit(filtered_user_likes)

total_recs = 0
users_with_recs = 0

for user_id in list(selected_users)[:100]:
    recommendations, _ = system.recommend(target_user=user_id)
    if recommendations:
        users_with_recs += 1
        total_recs += len(recommendations)

print(f"\nБазовая конфигурация (fail_limit=3, greedy):")
print(f"  Из 100 пользователей получили рекомендации: {users_with_recs}")
print(f"  Всего рекомендаций: {total_recs}")
print(f"  В среднем на пользователя: {total_recs / 100:.2f}")

print("ВЫВОДЫ ПО ЗАДАЧЕ 4")

print("\nЗадача 4 выполнена.")