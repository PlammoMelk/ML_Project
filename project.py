import pandas as pd
from scipy.io import arff

def load_arff_to_dataframe(file_path):
    """Load an ARFF file into a pandas DataFrame."""
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    
    # Decode byte strings if necessary
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    
    return df


def main():
    """Main function to load and display an ARFF file."""
    file_path = r"C:\\Users\\Owen\Documents\\ML_Project\\connect_4_data.arff"  # Update with your actual file path
    df = load_arff_to_dataframe(file_path)
    
    # Print the first few rows
    print(df.head()) 

if __name__ == "__main__":
    main()
