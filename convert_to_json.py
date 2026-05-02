import re

def parse_artifacts(ovis_output):
    """
    Parses the given Ovis output to extract artifact names and their corresponding explanations.
    
    Args:
        ovis_output (str): The raw Ovis output.
    
    Returns:
        list: A list of dictionaries, each containing an artifact name and its explanation.
        If parsing fails, logs the error and stores the failed output in a file.
    """

    artifact_explanation_list = []
    failed_parsing = []
    
    try:
        # Find all artifact name and explanation pairs from the Ovis output
        artifacts = re.findall(r'\*\*Artifact Name\*\*: (.*?)\n\*\*Explanation\*\*: (.*?)\n', ovis_output, re.DOTALL)
        
        # Build the explanation dictionary for easy conversion to JSON
        dictionary = {artifact[0]: artifact[1].strip() for artifact in artifacts}
        for artifact, explanation in dictionary.items():
          artifact_explanation_list.append({
              "Artifact": artifact,
              "Explanation": explanation
          })
    # Skips the JSON conversion if the Ovis output is not in a standard format
    except Exception as e:
        failed_parsing.append(f"Error: {str(e)}\nContent:\n{ovis_output}\n")
        print("Parsing failed!")
        with open("failed_parsing.txt", "a") as txt_file:
            txt_file.writelines(failed_parsing)

    return artifact_explanation_list


def json_formatting(filename, parsed_data, json_output):
    """
    Formats the parsed artifact data into a structured JSON format.
    
    Args:
        filename (str): The name of the image.
        parsed_data (list): A list of dictionaries containing parsed artifact data.
        json_output (list): A list where the formatted JSON data will be appended.
    
    Returns:
        list: The updated list containing the formatted JSON data.
    """
    
    image_index = int(filename[:-4])
    explanation_dict = {artifact['Artifact']: artifact['Explanation'] for artifact in parsed_data}

    json_output.append({
        "index": image_index,
        "explanation": explanation_dict
    })
    
    return json_output