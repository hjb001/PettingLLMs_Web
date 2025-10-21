import pandas as pd
import datasets

def read_search_data(file_path: str) -> pd.DataFrame:
    """
    Read search problem data from a CSV file or a Hugging Face dataset.

    Args:
        file_path: Path to the CSV file or Hugging Face dataset identifier.
    
    """
    df = pd.read_parquet(file_path)
    return df

data = read_search_data("/home/nvidia/user/junbo/PettingLLMs_Web/data/search/test/gaia_text_only.parquet")
print("数据形状:", data.shape)
print("\n第一行数据（索引0）:")
print(data.iloc[1]['Question'])


    