import sys  # Used to read token argument from command line
import json  # for saving and parsing json files

import numpy  # for generic operations
import pandas  # dumping json into csv
import sentence_transformers  # embeddings
import sklearn.decomposition  # orient tsne
import sklearn.manifold  # make a tsne
from numpy.typing import NDArray
import urllib.request
import datasets

# Get API token from command line
HF_TOKEN = sys.argv[1]

# Faculty lists
list_of_maps: list[tuple[str, str]] = [
    (
        "Mechanical Engineering",
        "https://raw.githubusercontent.com/cmccomb/map-of-mech/",
    ),
    (
        "Civil & Environmental Engineering",
        "https://raw.githubusercontent.com/cmccomb/map-of-civil/",
    ),
    (
        "Electrical & Computer Engineering",
        "https://raw.githubusercontent.com/cmccomb/map-of-ece/",
    ),
    (
        "Materials Science & Engineering",
        "https://raw.githubusercontent.com/cmccomb/map-of-mse/",
    ),
    ("Chemical Engineering", "https://raw.githubusercontent.com/cmccomb/map-of-cheme/"),
    (
        "Engineering & Public Policy",
        "https://raw.githubusercontent.com/cmccomb/map-of-epp/",
    ),
    ("Biomedical Engineering", "https://raw.githubusercontent.com/cmccomb/map-of-bme/"),
    ("CMU-Africa", "https://raw.githubusercontent.com/cmccomb/map-of-cmu-africa/"),
    (
        "CMU Silicon Valley",
        "https://raw.githubusercontent.com/cmccomb/map-of-cmu-silicon-valley/",
    ),
    (
        "Information Networking Institute",
        "https://raw.githubusercontent.com/cmccomb/map-of-ini/",
    ),
    (
        "Integrated Innovation Institute",
        "https://raw.githubusercontent.com/cmccomb/map-of-ini/",
    ),
]

FACULTY_EXTENSION: str = "main/faculty.csv"
JSON_EXTENSION: str = "main/data/"

all_names = []

# Dump all the json files into a single dataframe
all_the_data: pandas.DataFrame = pandas.DataFrame()
for map in list_of_maps:
    names = pandas.read_csv(map[1] + FACULTY_EXTENSION)["name"].to_list()
    ids = (
        pandas.read_csv(map[1] + FACULTY_EXTENSION)["id"]
        .replace(numpy.nan, None)
        .to_list()
    )
    for name, id in zip(names, ids):
        print(name, id)
        if id:
            encoded_name = name.replace(" ", "%20")
            json_url = map[1] + JSON_EXTENSION + encoded_name + ".json"
            all_names.append(name)

            with urllib.request.urlopen(json_url) as url:
                json_dict: dict = json.load(url)
                json_as_df: pandas.DataFrame = pandas.DataFrame.from_dict(json_dict)
                json_as_df["faculty"] = name
                json_as_df["department"] = map[0]
                all_the_data: pandas.DataFrame = pandas.concat(
                    [all_the_data, json_as_df], axis=0
                )

# Re-index the dataframe because it appears to eb necessary
all_the_data.reset_index(inplace=True)

# Embed titles from publications
model = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
embeddings: numpy.ndarray = model.encode(
    all_the_data["title"].to_list(), show_progress_bar=True
)

# Boil down teh data into a 2D plot
tsne_embeddings: NDArray = sklearn.manifold.TSNE(
    n_components=2, random_state=42, verbose=True
).fit_transform(embeddings)
pca_embeddings: NDArray = sklearn.decomposition.PCA(
    n_components=2, random_state=42
).fit_transform(tsne_embeddings)
all_the_data["x"] = pca_embeddings[:, 0]
all_the_data["y"] = pca_embeddings[:, 1]

all_the_data.to_csv("data.csv", index=False)


# Convert to a dataset and upload to huggingface. Converting to pandas and then to dataset avoids some weird errors
all_the_data["embedding"] = embeddings
publication_dataset = datasets.Dataset.from_pandas(all_the_data)
publication_dataset.push_to_hub("ccm/publications", token=HF_TOKEN)
