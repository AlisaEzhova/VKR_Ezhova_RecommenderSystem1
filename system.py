import random
from collections import defaultdict

class RecommenderSystem:
    """
    Рекомендательная система на основе логических правил (если-то).

    Шесть гипотез:
    H1 - пересечение плейлистов (основной механизм)
    H2 - приоритет по размеру пересечения (жадная сортировка)
    H3 - учёт сильных прослушиваний (дослушивание до конца)
    H4 - лимит отказов (сколько раз показывать треки от одного соседа)
    H5 - глобальное исключение дизлайков
    H6 - асимметричное влияние (опытный пользователь ценнее новичка)

    Каждая гипотеза реализована в отдельном методе.
    Параметры можно менять через config.
    """

    def __init__(self, config):
        """
        config: словарь с настройками системы.
        """
        # Основные параметры системы
        self.max_recommendations = config.get('max_recommendations', 10)  # сколько треков рекомендовать
        self.fail_limit = config.get('fail_limit', 3)  # лимит отказов (H4)
        self.strategy = config.get('strategy', 'greedy')  # greedy или random
        self.min_common = config.get('min_common', 1)  # минимальное количество общих треков (H1)
        self.max_neighbors = config.get('max_neighbors', None)  # ограничение на количество соседей

        # Флаги для включения/выключения гипотез
        # True - гипотеза активна, False - выключена
        self.use_H1 = config.get('use_H1', True)   # пересечение плейлистов
        self.use_H2 = config.get('use_H2', True)   # приоритет по размеру пересечения
        self.use_H3 = config.get('use_H3', False)  # учёт сильных прослушиваний
        self.use_H4 = config.get('use_H4', True)   # лимит отказов
        self.use_H5 = config.get('use_H5', False)  # исключение дизлайков
        self.use_H6 = config.get('use_H6', False)  # асимметричное влияние

        # Данные пользователей (заполняются в fit)
        self.user_likes = None
        self.user_strong = None
        self.user_dislikes = None

    def fit(self, user_likes, user_strong=None, user_dislikes=None):
        """
        Сохраняет данные пользователей.
        user_likes: словарь {id: множество треков, которые пользователь лайкнул}
        user_strong: словарь {id: множество треков, дослушанных до конца} (для H3)
        user_dislikes: словарь {id: множество треков, которые пользователь дизлайкнул} (для H5)
        """
        self.user_likes = user_likes
        self.user_strong = user_strong if user_strong else {}
        self.user_dislikes = user_dislikes if user_dislikes else {}
        return self

    def _hypothesis_H1_get_liked_tracks(self, user_id):
        """
        ГИПОТЕЗА H1: пересечение плейлистов.
        Возвращает треки, которые пользователь лайкнул.
        Это основа для поиска соседей.
        """
        return self.user_likes.get(user_id, set())

    def _hypothesis_H3_add_strong_listens(self, user_id):
        """
        ГИПОТЕЗА H3: учёт сильных прослушиваний.
        Если гипотеза включена, добавляет треки, которые пользователь дослушал до конца.
        Такие треки приравниваются к лайкам.
        """
        if self.use_H3:
            return self.user_strong.get(user_id, set())
        return set()

    def _get_all_user_tracks(self, user_id):
        """
        Объединяет H1 и H3.
        Возвращает множество всех треков, которые считаются "понравившимися"
        (лайки + сильные прослушивания, если H3 включена).
        """
        tracks = self._hypothesis_H1_get_liked_tracks(user_id)
        tracks.update(self._hypothesis_H3_add_strong_listens(user_id))
        return tracks

    def _hypothesis_H5_get_excluded_tracks(self, user_id):
        """
        ГИПОТЕЗА H5: глобальное исключение дизлайков.
        Если гипотеза включена, возвращает треки, которые пользователь дизлайкнул.
        Эти треки никогда не попадут в рекомендации для этого пользователя.
        """
        if self.use_H5:
            return self.user_dislikes.get(user_id, set())
        return set()

    def _hypothesis_H6_asymmetric_weight(self, neighbor_items, target_items):
        """
        ГИПОТЕЗА H6: асимметричное влияние.
        Если H6 выключена, возвращает 1.0 (вес не меняется).
        Если H6 включена:
        - Если у соседа больше треков, чем у целевого, вес увеличивается (сосед = эксперт)
        - Если у соседа меньше треков, вес уменьшается (сосед = новичок)
        """
        if not self.use_H6:
            return 1.0
        if len(neighbor_items) > len(target_items):
            return len(neighbor_items) / max(len(target_items), 1)
        else:
            return 0.5

    def _find_neighbors(self, target_user):
        """
        Находит пользователей, у которых есть общие треки с целевым (H1).
        Применяет асимметричный вес (H6).
        Ограничивает количество соседей (max_neighbors).
        """
        # Получаем треки целевого пользователя (с учётом H1 и H3)
        target_items = self._get_all_user_tracks(target_user)
        if not target_items:
            return []

        neighbors = []
        for user, likes in self.user_likes.items():
            if user == target_user:
                continue

            # Получаем треки соседа (с учётом H1 и H3)
            neighbor_items = self._get_all_user_tracks(user)
            # Количество общих треков
            intersection = len(target_items.intersection(neighbor_items))

            # H1: если есть общие треки
            if intersection >= self.min_common:
                # Начальный вес = количество общих треков
                weight = intersection
                # H6: применяем асимметричный вес
                asym_weight = self._hypothesis_H6_asymmetric_weight(neighbor_items, target_items)
                weight = weight * asym_weight

                neighbors.append({
                    'user': user,
                    'weight': weight,
                    'intersection': intersection,
                    'items': neighbor_items
                })

        # Ограничиваем количество соседей (если задано)
        if self.max_neighbors is not None and len(neighbors) > self.max_neighbors:
            # Сначала сортируем по весу, берём top-N
            neighbors.sort(key=lambda x: x['weight'], reverse=True)
            neighbors = neighbors[:self.max_neighbors]

        return neighbors

    def _hypothesis_H2_sort_neighbors(self, neighbors):
        """
        ГИПОТЕЗА H2: приоритет по размеру пересечения.
        greedy: сортировка по весу (количеству общих треков) по убыванию.
        random: случайное перемешивание.
        """
        if self.strategy == 'greedy':
            neighbors.sort(key=lambda x: x['weight'], reverse=True)
        elif self.strategy == 'random':
            random.shuffle(neighbors)
        return neighbors

    def _hypothesis_H4_check_fail_limit(self, target_user, neighbor_user, fail_counter):
        """
        ГИПОТЕЗА H4: лимит отказов.
        Проверяет, не превысил ли сосед лимит неудачных рекомендаций.
        Если H4 выключена, всегда возвращает True (можно рекомендовать).
        """
        if not self.use_H4:
            return True
        fail_count = fail_counter.get((target_user, neighbor_user), 0)
        return fail_count < self.fail_limit

    def recommend(self, target_user, fail_counter=None):
        """
        Формирует список рекомендаций для целевого пользователя.

        Порядок работы:
        1. Получить треки целевого пользователя (H1 + H3)
        2. Получить исключённые треки (H5)
        3. Найти соседей (H1, H6) и ограничить их количество (max_neighbors)
        4. Отсортировать соседей (H2)
        5. Для каждого соседа проверить лимит отказов (H4)
        6. Добавить новые треки в рекомендации
        """
        if fail_counter is None:
            fail_counter = defaultdict(int)

        # Шаг 1: треки целевого пользователя (H1 + H3)
        target_items = self._get_all_user_tracks(target_user)
        # Шаг 2: исключённые треки (H5)
        excluded_items = self._hypothesis_H5_get_excluded_tracks(target_user)

        # Шаг 3: поиск соседей (H1, H6)
        neighbors = self._find_neighbors(target_user)
        if not neighbors:
            return [], fail_counter

        # Шаг 4: сортировка соседей (H2)
        neighbors = self._hypothesis_H2_sort_neighbors(neighbors)

        recommendations = []
        seen = set()

        # Шаг 5: формирование рекомендаций
        for neighbor in neighbors:
            neighbor_user = neighbor['user']
            neighbor_items = neighbor['items']

            # H4: проверка лимита отказов
            if not self._hypothesis_H4_check_fail_limit(target_user, neighbor_user, fail_counter):
                continue

            # Кандидаты = треки соседа минус:
            # - уже понравившиеся целевому
            # - исключённые (дизлайки)
            # - уже добавленные в рекомендации
            candidates = neighbor_items - target_items - excluded_items - seen

            # Добавляем кандидатов в рекомендации
            for track in candidates:
                if len(recommendations) >= self.max_recommendations:
                    break
                recommendations.append(track)
                seen.add(track)

            if len(recommendations) >= self.max_recommendations:
                break

        return recommendations, fail_counter