from transformers import AutoModel
import PIL.Image as Image
import pandas as pd
from utils import common_parser
from constants import artifact_index_dict
model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)

def run_clip(image_path :str,object_class:int,dataframe:pd.DataFrame,clip_prediction_list_limit:int,threshold:int) -> list:
    """
    Run inference using the Jina-Clip-v2 model on a given image to retrieve the most probable artifacts based on similarity to a specific object class.

    Args:
        image_path (str): The file path to the input image.
        object_class (int): The predicted class label of the object in the image.
        dataframe (pd.DataFrame): A DataFrame containing the classification data and artifacts information.
        clip_prediction_list_limit (int): The maximum number of top artifact predictions to retrieve.
        threshold (float): The similarity threshold for selecting artifacts.

    Returns:
        list: A list of the top artifacts, sorted by similarity, based on the provided threshold and object class.
    """

    # Find the embeddings for the image and the artifacts using Jina-Clip-v2
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image_embeddings = model.encode_image(image)
    text_embeddings = model.encode_text(list(artifact_index_dict.keys()))

    # Parse the tsv and store the artifacts in three separate lists in order of the model's 
    # confidence of detecting the artifact correctly
    clip_accurate_list=common_parser(dataframe["CLIP Accurate"][object_class])
    clip_approximate_list=common_parser(dataframe["CLIP Approximate"][object_class])
    clip_miscellaneous_list=common_parser(dataframe["CLIP Miscellaneous"][object_class])

    # Check the similarity score for only those artifacts which can occur in the particular class
    # e.g. If the class is Airplane, ignore artifacts such as "Improper Fur Direction Flows"
    clip_total=set(clip_accurate_list+clip_approximate_list+clip_miscellaneous_list)
    clip_relevant_list=list(clip_total)
    results = []
    for i in range(len(artifact_index_dict.keys())):
        results.append(text_embeddings[i] @ image_embeddings.T)

    clip_results_list_sorted = sorted(enumerate(results), key=lambda x: x[1], reverse=True)
    clip_prediction = []

    # Find the top-k artifacts having the similarity score above the given threshold
    for artifact_index, similarity in clip_results_list_sorted:
        if artifact_index in clip_relevant_list:
            if similarity > threshold and len(clip_prediction) < clip_prediction_list_limit:
                clip_prediction.append(artifact_index)
            if len(clip_prediction) == clip_prediction_list_limit:
                break

    return clip_prediction

