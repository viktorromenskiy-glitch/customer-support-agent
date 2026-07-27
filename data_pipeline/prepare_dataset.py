"""
Шаг 1: локальная подготовка датасета Customer Support on Twitter.
Запустить у себя на компьютере (Dell Inspiron, 32GB RAM — более чем достаточно).

Этот скрипт сам найдёт файл датасета (в текущей папке или в подпапке twcs/)
и сам определит, CSV это или Excel — не нужно ничего переименовывать вручную.
"""
import pandas as pd
from pathlib import Path

# --- Настройки ---
TARGET_COMPANIES = ['AppleSupport', 'Uber_Support', 'SpotifyCares', 'AmazonHelp', 'Delta']
OUTPUT_FILE = '../twitter_support_filtered.csv'

# --- Автопоиск файла датасета ---
def find_dataset_file():
    here = Path('..')
    candidates = []
    for pattern in ['twcs.csv', 'twcs.xlsx', '**/twcs.csv', '**/twcs.xlsx']:
        candidates.extend(here.glob(pattern))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        print("ОШИБКА: файл twcs.csv / twcs.xlsx не найден.")
        print(f"Текущая папка: {here.resolve()}")
        print("Содержимое текущей папки и подпапок:")
        for p in here.rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                print(f"  {p}")
        raise FileNotFoundError("Не найден файл датасета. Проверьте список выше "
                                 "и, если файл называется иначе, укажите точный путь вручную.")
    chosen = candidates[0]
    print(f"Найден файл датасета: {chosen}")
    return chosen

dataset_path = find_dataset_file()

print("Загрузка полного датасета...")
if dataset_path.suffix.lower() == '.xlsx' or dataset_path.suffix.lower() == '.xls':
    df = pd.read_excel(dataset_path)
else:
    df = pd.read_csv(dataset_path)
print(f"Всего строк: {len(df):,}")
print(f"Колонки: {list(df.columns)}")

# --- 1. Только полные диалоги (есть и inbound, и outbound) ---
df['is_company'] = ~df['author_id'].astype(str).str.match(r'^\d+$')

# --- 2. Фильтр по целевым компаниям ---
company_ids = df.loc[df['is_company'], 'author_id'].unique()
target_ids = [c for c in company_ids if c in TARGET_COMPANIES]
print(f"\nНайдены целевые компании в данных: {target_ids}")

df_companies = df[df['author_id'].isin(TARGET_COMPANIES)].copy()
print(f"Сообщений от целевых компаний: {len(df_companies):,}")

# --- 3. Восстанавливаем полные пары (запрос клиента -> ответ компании) ---
df_indexed = df.set_index('tweet_id')
pairs = []
for _, row in df_companies.iterrows():
    if pd.notna(row['in_response_to_tweet_id']):
        parent_id = row['in_response_to_tweet_id']
        if parent_id in df_indexed.index:
            parent = df_indexed.loc[parent_id]
            if isinstance(parent, pd.Series) and parent['inbound']:
                pairs.append({
                    'company': row['author_id'],
                    'request_text': parent['text'],
                    'response_text': row['text'],
                    'request_created_at': parent['created_at'],
                    'response_created_at': row['created_at'],
                    'request_tweet_id': parent_id,
                    'response_tweet_id': row['tweet_id'],
                })

df_pairs = pd.DataFrame(pairs)
print(f"\nВосстановлено полных пар (запрос+ответ): {len(df_pairs):,}")

# --- 4. Базовая статистика по длине (для эмпирического подбора порогов) ---
df_pairs['request_len'] = df_pairs['request_text'].str.len()
df_pairs['response_len'] = df_pairs['response_text'].str.len()
print(f"\nРаспределение длины запроса: min={df_pairs['request_len'].min()}, "
      f"p10={df_pairs['request_len'].quantile(0.1):.0f}, "
      f"median={df_pairs['request_len'].median():.0f}, max={df_pairs['request_len'].max()}")
print(f"Распределение длины ответа: min={df_pairs['response_len'].min()}, "
      f"p10={df_pairs['response_len'].quantile(0.1):.0f}, "
      f"median={df_pairs['response_len'].median():.0f}, max={df_pairs['response_len'].max()}")

# --- 5. Убираем дубликаты ---
before = len(df_pairs)
df_pairs = df_pairs.drop_duplicates(subset=['request_text', 'response_text'])
print(f"\nУдалено дубликатов: {before - len(df_pairs):,}")

# --- Сохранение ---
df_pairs.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Сохранено {len(df_pairs):,} пар в {OUTPUT_FILE}")
print(f"  Размер файла должен быть небольшим (несколько МБ) — можно будет загрузить в чат")
