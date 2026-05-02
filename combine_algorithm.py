from utils import *
import pandas as pd

def run_combination_algorithm(dataframe:pd.DataFrame,object_class:int,clip_list:list,cnn_dict:dict,n,limit_flag :int=0) ->list :
  """
  Returns the final list of predicted artifacts after combining the results from CNNs and CLIP using the combination algorithm.

  Args:
      dataframe (pd.Dataframe): A DataFrame containing the classification data and artifacts information.
      object_class (int): The predicted class label of the object in the image.
      clip_list (list): Sorted list of the artifact indices predicted by CLIP
      cnn_dict (dict): Dictionary containing the artifact indices along with their prediction from the CNNs.
      n (int): Determines the length of the final prediction list
      limit_flag (int, optional): Determines whether semantically similar classes account towards the prediction limit or not.

  Returns:
      list: The indices of the artifacts predicted by the model
  """

  predictions=[]
  class_specific_artifact_definite_dict, \
      class_specific_artifact_likely_dict, \
        class_specific_artifact_probable_dict =class_specific_parser(dataframe["Class Specific"][object_class])
  class_cnn_dict=cnn_parser(dataframe["CNN"][object_class])
  clip_accurate_list=common_parser(dataframe["CLIP Accurate"][object_class])
  clip_approximate_list=common_parser(dataframe["CLIP Approximate"][object_class])
  clip_miscellaneous_list=common_parser(dataframe["CLIP Miscellaneous"][object_class])

  # Appends the artifacts which have a very high chance of appearing in the object class 
  # along with the other semantically similar artifacts
  # e.g. Improper Fur Direction Flows in Cats
  for artifact_index in class_specific_artifact_definite_dict.keys():
    predictions.append(artifact_index)
    if class_specific_artifact_definite_dict[artifact_index]!=-1:
      for similar_artifact_index in class_specific_artifact_definite_dict[artifact_index]:
        predictions.append(similar_artifact_index)
        if not limit_flag:
          n+=1

  # Appends the artifacts which are detected by the CNN models
  for artifact_index in class_cnn_dict.keys():
    if cnn_dict[artifact_index]==1:
      predictions.append(artifact_index)
      if class_cnn_dict[artifact_index]!=-1:
        for similar_artifact_index in class_cnn_dict[artifact_index]:
          predictions.append(similar_artifact_index)
          if not limit_flag:
            n+=1

  # Appends the artifacts which CLIP detects with high confidence
  for artifact_index in clip_accurate_list:
    if artifact_index in clip_list:
      predictions.append(artifact_index)

  # Appends the artifacts which have a high chance of appearing in the object class and have been detected by CLIP
  # along with the other semantically similar artifacts
  # e.g. Glow or Light Bleed Around Object Boundaries in Cats
  for artifact_index in class_specific_artifact_likely_dict.keys():
    if artifact_index in clip_list:
      predictions.append(artifact_index)
      if class_specific_artifact_likely_dict[artifact_index]!=-1:
        for similar_artifact_index in class_specific_artifact_likely_dict[artifact_index]:
          predictions.append(similar_artifact_index)
          if not limit_flag:
            n+=1

  # Appends the artifacts which CLIP detects with sligthly less confidence
  for artifact_index in clip_approximate_list:
    if artifact_index in clip_list:
      predictions.append(artifact_index)

  # Appends the artifacts which have a good chance of appearing in the object class and have been detected by CLIP
  # along with the other semantically similar artifacts
  # e.g. Unnaturally Glossy Surfaces in Airplane
  for artifact_index in class_specific_artifact_probable_dict.keys():
    if artifact_index in clip_list:
      predictions.append(artifact_index)
      if class_specific_artifact_probable_dict[artifact_index]!=-1:
        for similar_artifact_index in class_specific_artifact_probable_dict[artifact_index]:
          predictions.append(similar_artifact_index)
          if not limit_flag:
            n+=1

  # Appends the artifacts which CLIP detects with lesser confidence
  for artifact_index in clip_miscellaneous_list:
    if artifact_index in clip_list:
      predictions.append(artifact_index)
      
  return predictions[:n]
  