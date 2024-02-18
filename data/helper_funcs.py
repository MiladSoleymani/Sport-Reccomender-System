import os
import json
import pandas as pd


def merge_parameters(folder_path: str) -> pd.DataFrame:
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter for files with a ".json" extension
    json_files = [file for file in files if file.endswith(".json")]

    all_data = []
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        # Load JSON data from file
        try:
            with open(json_path, "r") as json_file:
                json_data = json_file.read()

            # Check if the JSON data is not empty
            if not json_data:
                raise ValueError("The JSON file is empty.")

            # Parse JSON data
            data = json.loads(json_data)

            all_data.append(data)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Convert the dictionary to a Pandas DataFrame
    all_data = pd.DataFrame(all_data)

    # Convert the data type of a column in a pandas DataFrame from object to int
    all_data["id"] = all_data["id"].astype(int)

    return all_data


def merge_labels(folder_path: str, num_top_spotrts: int = 5) -> pd.DataFrame:
    # sports' names
    names = [
        "Badminton",
        "Basketball",
        "Boxing",
        "Baseball",
        "Tennis",
        "Taekwondo",
        "Pistol Shooting",
        "Archery",
        "Judo",
        "Athletics (sprint)",
        "Athletics (endurance)",
        "Athletics (throwing)",
        "Athletics (jumps)",
        "Cycling (road)",
        "Cycling (sprint)",
        "Softball",
        "Rock Climbing (indoor)",
        "Rock Climbing (speed)",
        "Swimming (long-distance)",
        "Swimming (sprint)",
        "Diving",
        "American Football",
        "Soccer",
        "Futsal",
        "Canoeing",
        "Lacrosse",
        "Field Hockey",
        "Ice Hockey",
        "Handball",
        "Water Polo",
        "Volleyball",
        "Weightlifting",
        "Wushu",
        "Rugby",
        "Table Tennis",
        "Gymnastics",
        "Karate",
        "Curling",
        "Cricket",
        "Wrestling",
        "Golf",
        "Fencing",
        "Rowing (long-distance)",
    ]

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter for files with a ".json" extension
    json_files = [file for file in files if file.endswith(".json")]

    all_data = []

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        # Load JSON data from file
        try:
            with open(json_path, "r") as json_file:
                json_data = json_file.read()

            # Check if the JSON data is not empty
            if not json_data:
                raise ValueError("The JSON file is empty.")

            # Parse JSON data
            data = json.loads(json_data)

            data_dict = {}
            data_dict["id"] = int(data["id"])

            for idx, _ in enumerate(data["sports"]):
                data_dict[f"sport-{idx}"] = _["main_name"]

                try:
                    data_dict[f"sport-{idx}"] = (
                        data_dict[f"sport-{idx}"]
                        + " "
                        + _["sub_name"].replace("(", "").replace(")", "")
                    )
                except:
                    pass

            all_data.append(data_dict)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Convert the dictionary to a Pandas DataFrame
    all_data = pd.DataFrame(all_data)

    # Convert the data type of a column in a pandas DataFrame from object to int
    all_data["id"] = all_data["id"].astype(int)

    # Create a new dataframe with 0s and 1s based on the presence of sports names
    result_data = {
        name: [
            1 if name in row else 0
            for row in all_data[
                [f"sport-{idx}" for idx in range(num_top_spotrts)]
            ].values
        ]
        for name in names
    }
    result_df = pd.DataFrame(result_data)

    # Combine ID column with the result dataframe
    result_df.insert(0, "id", all_data["id"])

    return result_df


def merge(parameters: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(
        parameters, labels, on="id", how="inner"
    )  # 'inner' means it will keep only the common rows


def clean(merge_data: pd.DataFrame) -> pd.DataFrame:
    return merge_data.dropna(axis=1)  # Drop columns containing NaN values


def set_digit_ratio(clean_data: pd.DataFrame) -> pd.DataFrame:
    # Replace 'E' with '3' in the 'Digit_Ratio' column
    clean_data["Digit_Ratio"] = clean_data["Digit_Ratio"].replace("E", 3)
    return clean_data
