import re
import pandas as pd

def separate_bracket_content(input_string: str) -> tuple:
  """Seperate the part of the string inside and outside the given input string 

  Args:
      input_string (str): The input string 

  Returns:
      tuple: Returns the part inside and outside the brackets as a tuple as present
  """
  match = re.match(r"^(.*?)(\((.*?)\))?$", input_string)

  if match:
      outside_bracket = match.group(1).strip()
      inside_bracket = match.group(3) if match.group(3) else None
      return outside_bracket, inside_bracket
  else:
      return input_string, None
    
def class_specific_parser(input_string) -> tuple:
  """ The parser for the value inside the Class Specific field defined in the Dataframe

  Args:
      input_string (str): The value inside the Special class

  Returns:
      tuple: Returns Three dictionaries that have the artifact indices and their semantically similar 
      artifact indices as key value pairs in order of their probablity of being present in the class.
  """
  input_string=str(input_string)

  if str(input_string) == "nan":
    return dict(),dict(),dict()

  class_specific_artifacts=list(input_string.split(','))
  definite_artifacts_dict=dict()
  likely_artifacts_dict=dict()
  probable_artifacts_dict=dict()

  for artifact in class_specific_artifacts:
    priority=int(artifact[-1])
    artifact_index, similar_artifacts_indices=separate_bracket_content(artifact[:-2])
    similar_artifacts_idx_list = [int(x) for x in similar_artifacts_indices.split(",")] if similar_artifacts_indices else -1
    if priority == 0:
        definite_artifacts_dict[int(artifact_index)]=similar_artifacts_idx_list
    elif priority == 1:
        likely_artifacts_dict[int(artifact_index)]=similar_artifacts_idx_list
    else:
        probable_artifacts_dict[int(artifact_index)]=similar_artifacts_idx_list
        
  return definite_artifacts_dict, likely_artifacts_dict, probable_artifacts_dict
    
  

def cnn_parser(input_string: str) -> dict:
  """The parser for the value inside the CNN class defined in the Dataframe.

  Args:
      input_string (str): The value inside the CNN class.

  Returns:
      dict: Returns the dictionary that has the artifact indices and their semantically similar artifact 
      indices as key value pairs.
  """
  cnn_artifacts=list(input_string.split(','))
  class_cnn_dict=dict()
  for artifact in cnn_artifacts:
      artifact_index, similar_artifacts_indices=separate_bracket_content(artifact)
      if similar_artifacts_indices:
        class_cnn_dict[int(artifact_index)]=[int(x) for x in similar_artifacts_indices.split(":")]
      else:
        class_cnn_dict[int(artifact_index)]=-1

  return class_cnn_dict


def common_parser(input_string: str) -> list:
  """The parser for the value inside the rest of the classes defined in the Dataframe.

  Args:
      input_string (str): The value inside the classes

  Returns:
      list: Returns the list of Artifact indices
  """
  artifact_indices=list(input_string.split(','))
  for i in range(len(artifact_indices)):
    artifact_indices[i]=int(artifact_indices[i])
  return artifact_indices