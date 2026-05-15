"""
experiment_task3.py

ЗАДАЧА 3
Исследование динамики популярности треков
и проверка гипотезы «хищник-жертва»
на основе временных сегментов потока событий
"""

from datasets import load_dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

# НАСТРОЙКИ ГРАФИКОВ ПО ГОСТ
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '-'

# НАСТРОЙКИ ЭКСПЕРИМЕНТА
SEGMENTS = 20
MIN_LIKES = 30

SAVE_FILE = 'event_timeline_data.pkl'


# ЧАСТЬ 1: ЗАГРУЗКА ИЛИ ВОССТАНОВЛЕНИЕ ДАННЫХ

print("ЗАДАЧА 3: АНАЛИЗ ХИЩНИК-ЖЕРТВА")

if os.path.exists(SAVE_FILE):
    print("\n[1/5] Загрузка сохранённых данных...")
    with open(SAVE_FILE, 'rb') as f:
        saved_data = pickle.load(f)
    
    organic_timeline = saved_data['organic_timeline']
    rec_timeline = saved_data['rec_timeline']
    total_events = saved_data['total_events']
    print(f"  Загружено органических треков: {len(organic_timeline)}")
    print(f"  Загружено рекомендационных треков: {len(rec_timeline)}")
    print(f"  Всего событий: {total_events}")
    print("  [OK] Данные загружены из кэша")

else:
    print("\n[1/5] Загрузка датасета (первый запуск, потребуется 20-40 минут)...")
    print("  Загружается файл multi_event.parquet (384 MB)...")
    
    multi_ds = load_dataset(
        "yandex/yambda",
        data_dir="flat/50m",
        data_files="multi_event.parquet",
        split="train"
    )
    
    print("  [OK] Файл загружен, обработка событий...")
    
    organic_timeline = defaultdict(list)
    rec_timeline = defaultdict(list)
    
    # Используем tqdm для прогресс-бара
    for idx, example in enumerate(tqdm(multi_ds, desc="  Обработка событий", unit=" событий")):
        if example['event_type'] != 'like':
            continue
        
        item = example['item_id']
        
        if example['is_organic']:
            organic_timeline[item].append(idx)
        else:
            rec_timeline[item].append(idx)
    
    total_events = idx + 1
    print(f"  Всего обработано событий: {total_events}")
    print(f"  Органических треков: {len(organic_timeline)}")
    print(f"  Рекомендационных треков: {len(rec_timeline)}")
    
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump({
            'organic_timeline': organic_timeline,
            'rec_timeline': rec_timeline,
            'total_events': total_events
        }, f)
    print("  [OK] Данные сохранены для быстрой загрузки в следующий раз")


# ЧАСТЬ 2: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def moving_average(data, window=3):
    """Сглаживание ряда скользящим средним"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='same')


def build_segment_series(timeline, total_events, segments=SEGMENTS):
    """Разбивает временную линию на сегменты и считает количество событий в каждом"""
    segment_size = total_events // segments if segments > 0 else 1
    series = []
    
    for seg in range(segments):
        left = seg * segment_size
        right = (seg + 1) * segment_size if seg < segments - 1 else total_events
        count = sum(1 for t in timeline if left <= t < right)
        series.append(count)
    
    return series


def detect_oscillations(series):
    """Определяет количество осцилляций (смен знака производной)"""
    diffs = np.diff(series)
    if len(diffs) == 0:
        return 0
    
    signs = np.sign(diffs)
    changes = 0
    
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1
    
    return changes


def calculate_correlation(series1, series2):
    """Вычисляет коэффициент корреляции Пирсона между двумя рядами"""
    if np.std(series1) == 0 or np.std(series2) == 0:
        return 0
    return np.corrcoef(series1, series2)[0, 1]


# ЧАСТЬ 3: АНАЛИЗ ТОП-10 ТРЕКОВ

# ЧАСТЬ 3: АНАЛИЗ ТОП-10 ТРЕКОВ (РИСУНКИ 3.1a И 3.1b)

print("\n[2/5] Анализ топ-10 треков...")

item_total = defaultdict(int)
for item, tl in organic_timeline.items():
    item_total[item] += len(tl)
for item, tl in rec_timeline.items():
    item_total[item] += len(tl)

top_items = sorted(item_total.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"  Выбрано 10 наиболее популярных треков")

# Функция для построения одного ряда графиков (5 штук)
def plot_tracks_subset(items_subset, start_idx, filename, title):
    """Построить график для 5 треков в одной строке"""
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))  # 1 строка, 5 столбцов
    axes = axes.flatten()
    
    for idx, (item_id, likes) in enumerate(items_subset):
        ax = axes[idx]
        
        timeline = []
        if item_id in organic_timeline:
            timeline += organic_timeline[item_id]
        if item_id in rec_timeline:
            timeline += rec_timeline[item_id]
        
        series = build_segment_series(timeline, total_events, SEGMENTS)
        smooth_series = moving_average(series)
        oscillations = detect_oscillations(smooth_series)
        
        ax.plot(smooth_series, linewidth=2, color='blue')
        ax.set_title(f'Трек {item_id}\nлайков: {likes}\nосцилляций: {oscillations}', fontsize=10)
        ax.set_xlabel('Временной сегмент, н.е.', fontsize=9)
        ax.set_ylabel('Популярность, лайки', fontsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Закрываем фигуру, чтобы не накапливались
    print(f"  [OK] График сохранён как {filename}")

# Построение первого графика (треки 1-5)
plot_tracks_subset(
    top_items[:5], 
    start_idx=0, 
    filename='figure_3_1a.png', 
    title='Динамика популярности треков 1–5 (наиболее популярные)'
)

# Построение второго графика (треки 6-10)
plot_tracks_subset(
    top_items[5:10], 
    start_idx=5, 
    filename='figure_3_1b.png', 
    title='Динамика популярности треков 6–10'
)

print("  [OK] Оба графика сохранены (figure_3_1a.png и figure_3_1b.png)")


# ЧАСТЬ 4: АНАЛИЗ ВСЕХ ТРЕКОВ (ПОИСК ХИЩНИКОВ, ЖЕРТВ, ОСЦИЛЛЯЦИЙ)

print("\n[3/5] Анализ динамики всех треков...")

track_data = {}

print("  Обработка органических треков...")
for item, timeline in tqdm(organic_timeline.items(), desc="    Прогресс"):
    if len(timeline) < MIN_LIKES:
        continue
    
    timeline.sort()
    series = build_segment_series(timeline, total_events, SEGMENTS)
    smooth_series = moving_average(series)
    
    growth = smooth_series[-1] - smooth_series[0]
    oscillations = detect_oscillations(smooth_series)
    
    track_data[item] = {
        'series': smooth_series,
        'growth': growth,
        'oscillations': oscillations,
        'likes': len(timeline),
        'type': 'organic'
    }

print("  Обработка рекомендационных треков...")
for item, timeline in tqdm(rec_timeline.items(), desc="    Прогресс"):
    if len(timeline) < MIN_LIKES:
        continue
    
    timeline.sort()
    series = build_segment_series(timeline, total_events, SEGMENTS)
    smooth_series = moving_average(series)
    
    growth = smooth_series[-1] - smooth_series[0]
    oscillations = detect_oscillations(smooth_series)
    
    if item in track_data:
        track_data[item]['likes'] += len(timeline)
        track_data[item]['series'] = [(track_data[item]['series'][i] + smooth_series[i]) for i in range(len(smooth_series))]
    else:
        track_data[item] = {
            'series': smooth_series,
            'growth': growth,
            'oscillations': oscillations,
            'likes': len(timeline),
            'type': 'recommended'
        }

predators = [(item, data) for item, data in track_data.items() if data['growth'] >= 3]
prey = [(item, data) for item, data in track_data.items() if data['growth'] <= -3]
oscillating = [(item, data) for item, data in track_data.items() if data['oscillations'] >= 5]

predators.sort(key=lambda x: -x[1]['growth'])
prey.sort(key=lambda x: x[1]['growth'])
oscillating.sort(key=lambda x: -x[1]['oscillations'])

print(f"\n  Результаты анализа:")
print(f"    Всего треков с >= {MIN_LIKES} лайками: {len(track_data)}")
print(f"    Хищников (рост >= 3): {len(predators)}")
print(f"    Жертв (падение <= -3): {len(prey)}")
print(f"    Треков с осцилляциями (>= 5): {len(oscillating)}")


# ЧАСТЬ 5: ПОИСК ПАР ХИЩНИК-ЖЕРТВА

print("\n[4/5] Поиск пар хищник-жертва...")

pairs = []

for pred_item, pred_data in predators[:10]:
    for prey_item, prey_data in prey[:10]:
        corr = calculate_correlation(pred_data['series'], prey_data['series'])
        pairs.append({
            'predator': pred_item,
            'prey': prey_item,
            'predator_growth': round(pred_data['growth'], 2),
            'prey_growth': round(prey_data['growth'], 2),
            'predator_osc': pred_data['oscillations'],
            'prey_osc': prey_data['oscillations'],
            'correlation': round(corr, 2)
        })

if pairs:
    print(f"  Найдено пар: {len(pairs)}")
    
    df = pd.DataFrame(pairs[:15])
    print("\n  ТАБЛИЦА ПАР ХИЩНИК-ЖЕРТВА\n")
    print(df.to_string(index=False))
    
    df.to_csv('predator_prey_pairs_table.csv', index=False)
    print("\n  [OK] Таблица сохранена как predator_prey_pairs_table.csv")
else:
    print("  Пар хищник-жертва не обнаружено")


# ЧАСТЬ 6: ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ ПАР (РИСУНОК 3.2)

print("\n[5/5] Построение графиков...")

os.makedirs('predator_prey_pairs', exist_ok=True)

if pairs:
    # Берём первую пару для основного графика
    pair = pairs[0]
    print(f"  Построение графика для пары {pair['predator']} - {pair['prey']}...")
    
    pred_series = None
    prey_series = None
    
    for item, data in predators:
        if item == pair['predator']:
            pred_series = data['series']
            break
    
    for item, data in prey:
        if item == pair['prey']:
            prey_series = data['series']
            break
    
    if pred_series is not None and prey_series is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(pred_series, linewidth=2, color='red', label=f'Хищник (трек {pair["predator"]})')
        ax.plot(prey_series, linewidth=2, color='blue', label=f'Жертва (трек {pair["prey"]})')
        
        ax.set_title(f'Пара хищник-жертва\nкорреляция = {pair["correlation"]}', fontsize=12)
        ax.set_xlabel('Временной сегмент, н.е.', fontsize=12)
        ax.set_ylabel('Популярность, лайки', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('figure_3_2.png', dpi=300, bbox_inches='tight')
        print("  [OK] Основной график сохранён как figure_3_2.png")
    
    # Сохраняем все пары в папку
    print("  Сохранение графиков всех пар...")
    for idx, pair in enumerate(tqdm(pairs[:10], desc="    Прогресс")):
        pred_series = None
        prey_series = None
        
        for item, data in predators:
            if item == pair['predator']:
                pred_series = data['series']
                break
        
        for item, data in prey:
            if item == pair['prey']:
                prey_series = data['series']
                break
        
        if pred_series is not None and prey_series is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(pred_series, linewidth=2, color='red', label=f'Хищник (трек {pair["predator"]})')
            ax.plot(prey_series, linewidth=2, color='blue', label=f'Жертва (трек {pair["prey"]})')
            ax.set_xlabel('Временной сегмент, н.е.', fontsize=10)
            ax.set_ylabel('Популярность, лайки', fontsize=10)
            ax.set_title(f'Корреляция = {pair["correlation"]}', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'predator_prey_pairs/pair_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  [OK] Сохранено {min(10, len(pairs))} графиков пар в папку predator_prey_pairs/")

else:
    # Если пар нет, строим график подозрительных треков
    print("  Построение графика подозрительных треков...")
    
    suspicious_tracks = (predators + oscillating)[:3]
    fig, axes = plt.subplots(1, len(suspicious_tracks), figsize=(15, 5))
    if len(suspicious_tracks) == 1:
        axes = [axes]
    
    for idx, (item, data) in enumerate(suspicious_tracks):
        ax = axes[idx]
        ax.plot(data['series'], linewidth=2, color='blue')
        ax.set_title(f'Трек {item}\nлайков: {data["likes"]}\nрост: {data["growth"]}', fontsize=10)
        ax.set_xlabel('Временной сегмент, н.е.', fontsize=9)
        ax.set_ylabel('Популярность, лайки', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Треки с аномальной динамикой', fontsize=14)
    plt.tight_layout()
    plt.savefig('figure_3_2.png', dpi=300, bbox_inches='tight')
    print("  [OK] График сохранён как figure_3_2.png")


# ЧАСТЬ 7: ЗАВЕРШЕНИЕ

print("ЗАДАЧА 3 ВЫПОЛНЕНА")
print("\nСохранённые файлы:")
print("  - figure_3_1.png (топ-10 треков)")
print("  - figure_3_2.png (основной график пар или аномальных треков)")
print("  - predator_prey_pairs_table.csv (таблица пар)")
if pairs:
    print(f"  - predator_prey_pairs/ ({min(10, len(pairs))} графиков пар)")
else:
    print("  - predator_prey_pairs/ (пусто, пары не найдены)")
print("\n")