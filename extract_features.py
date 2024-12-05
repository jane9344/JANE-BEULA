import pandas as pd

def extract_features(filepath):
    """
    Extract features from a Solidity contract file.
    
    Features include:
    - Length of the code
    - Counts of specific keywords like function, modifier, selfdestruct, require, and fallback.

    Args:
    - filepath: Path to the Solidity contract file

    Returns:
    - DataFrame: A DataFrame containing extracted features (used for prediction)
    """

    # Open and read the contract file
    with open(filepath, 'r', encoding='utf-8') as file:
        code = file.read()

    # Extract features from the code
    features = {
        'length': len(code),  # Length of the contract in characters
        'function_count': code.count('function'),  # Count of 'function' keyword
        'modifier_count': code.count('modifier'),  # Count of 'modifier' keyword
        'selfdestruct_count': code.count('selfdestruct'),  # Count of 'selfdestruct' keyword
        'require_count': code.count('require'),  # Count of 'require' keyword
        'fallback_count': code.count('fallback')  # Count of 'fallback' keyword
    }
    
    # Convert the features to a pandas DataFrame (ensure correct column order)
    features_df = pd.DataFrame([features])  # Return as DataFrame with one row of features
    
    # Reorder columns to ensure they match the trained model's expected order
    features_df = features_df[['length', 'function_count', 'modifier_count', 'selfdestruct_count', 'require_count', 'fallback_count']]
    
    return features_df
