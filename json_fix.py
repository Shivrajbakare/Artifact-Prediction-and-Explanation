import json
import argparse

artifact_index_dict_lowercase = {
    "abruptly cut off objects": "Abruptly cut off objects",
    "aliasing along high-contrast edges": "Aliasing along high-contrast edges",
    "anatomically impossible joint configurations": "Anatomically impossible joint configurations",
    "anatomically incorrect paw structures": "Anatomically incorrect paw structures",
    "artificial depth of field in object presentation": "Artificial depth of field in object presentation",
    "artificial enhancement artifacts": "Artificial enhancement artifacts",
    "artificial noise patterns in uniform surfaces": "Artificial noise patterns in uniform surfaces",
    "artificial smoothness": "Artificial smoothness",
    "asymmetric features in naturally symmetric objects": "Asymmetric features in naturally symmetric objects",
    "biological asymmetry errors": "Biological asymmetry errors",
    "blurred boundaries in fine details": "Blurred boundaries in fine details",
    "cinematization effects": "Cinematization Effects",
    "color coherence breaks": "Color coherence breaks",
    "dental anomalies in mammals": "Dental anomalies in mammals",
    "depth perception anomalies": "Depth perception anomalies",
    "discontinuous surfaces": "Discontinuous surfaces",
    "distorted window reflections": "Distorted window reflections",
    "dramatic lighting that defies natural physics": "Dramatic lighting that defies natural physics",
    "exaggerated characteristic features": "Exaggerated characteristic features",
    "excessive sharpness in certain image regions": "Excessive sharpness in certain image regions",
    "fake depth of field": "Fake depth of field",
    "floating or disconnected components": "Floating or disconnected components",
    "frequency domain signatures": "Frequency domain signatures",
    "ghosting effects: semi-transparent duplicates of elements": "Ghosting effects: Semi-transparent duplicates of elements",
    "glow or light bleed around object boundaries": "Glow or light bleed around object boundaries",
    "implausible aerodynamic structures": "Implausible aerodynamic structures",
    "impossible foreshortening in animal bodies": "Impossible foreshortening in animal bodies",
    "impossible mechanical connections": "Impossible mechanical connections",
    "impossible mechanical joints": "Impossible mechanical joints",
    "improper fur direction flows": "Improper fur direction flows",
    "inconsistent material properties": "Inconsistent material properties",
    "inconsistent object boundaries": "Inconsistent object boundaries",
    "inconsistent scale of mechanical parts": "Inconsistent scale of mechanical parts",
    "inconsistent shadow directions": "Inconsistent shadow directions",
    "incorrect perspective rendering": "Incorrect perspective rendering",
    "incorrect reflection mapping": "Incorrect reflection mapping",
    "incorrect skin tones": "Incorrect Skin Tones",
    "incorrect wheel geometry": "Incorrect wheel geometry",
    "irregular proportions in mechanical components": "Irregular proportions in mechanical components",
    "jagged edges in curved structures": "Jagged edges in curved structures",
    "loss of fine detail in complex structures": "Loss of fine detail in complex structures",
    "metallic surface artifacts": "Metallic surface artifacts",
    "misaligned bilateral elements in animal faces": "Misaligned bilateral elements in animal faces",
    "misaligned body panels": "Misaligned body panels",
    "misshapen ears or appendages": "Misshapen ears or appendages",
    "missing ambient occlusion": "Missing ambient occlusion",
    "movie-poster like composition of ordinary scenes": "Movie-poster like composition of ordinary scenes",
    "multiple inconsistent shadow sources": "Multiple inconsistent shadow sources",
    "multiple light source conflicts": "Multiple light source conflicts",
    "non-manifold geometries in rigid structures": "Non-manifold geometries in rigid structures",
    "over-sharpening artifacts": "Over-sharpening artifacts",
    "over-smoothing of natural textures": "Over-smoothing of natural textures",
    "physically impossible structural elements": "Physically impossible structural elements",
    "random noise patterns in detailed areas": "Random noise patterns in detailed areas",
    "regular grid-like artifacts in textures": "Regular grid-like artifacts in textures",
    "repeated element patterns": "Repeated element patterns",
    "resolution inconsistencies within regions": "Resolution inconsistencies within regions",
    "scale inconsistencies within single objects": "Scale inconsistencies within single objects",
    "scale inconsistencies within the same object class": "Scale inconsistencies within the same object class",
    "spatial relationship errors": "Spatial relationship errors",
    "synthetic material appearance": "Synthetic material appearance",
    "systematic color distribution anomalies": "Systematic color distribution anomalies",
    "texture bleeding between adjacent regions": "Texture bleeding between adjacent regions",
    "texture repetition patterns": "Texture repetition patterns",
    "unnatural color transitions": "Unnatural color transitions",
    "unnatural lighting gradients": "Unnatural Lighting Gradients",
    "unnatural pose artifacts": "Unnatural pose artifacts",
    "unnaturally glossy surfaces": "Unnaturally glossy surfaces",
    "unrealistic eye reflections": "Unrealistic eye reflections",
    "unrealistic specular highlights": "Unrealistic specular highlights"
}

parser = argparse.ArgumentParser(description="Update explanation keys in a JSON file using a predefined mapping.")
parser.add_argument(
    "--input_file", "-i", required=True, help="Path to the input JSON file containing the data."
)
parser.add_argument(
    "--output_file", "-o", required=True, help="Path to the output JSON file to save the updated data."
)
args = parser.parse_args()


with open(args.input_file, "r") as infile:
    data = json.load(infile)

for item in data:
    explanation = item.get("explanation", {})
    updated_explanation = {}
    for key, value in explanation.items():
        updated_key = artifact_index_dict_lowercase.get(key.lower(), key)
        updated_explanation[updated_key] = value
    item["explanation"] = updated_explanation

with open(args.output_file, "w") as outfile:
    json.dump(data, outfile, indent=4)

print(f"Updated JSON saved to {args.output_file}")