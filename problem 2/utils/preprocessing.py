import pandas as pd

def load_and_prepare_dataset(path):
    df = pd.read_csv(path)

    # Select useful columns
    df = df[['Review', 'Recommended']].dropna()

    # Normalize labels
    df['Recommended'] = df['Recommended'].astype(str).str.lower()
    df['label'] = df['Recommended'].apply(
        lambda x: 'positive' if x in ['yes', '1', 'true'] else 'negative'
    )

    return list(zip(df['Review'], df['label']))
