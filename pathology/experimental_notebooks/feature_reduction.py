import json 
import umap

with open("./embedded_image_collection.txt") as f:
    embedded_images = f.readlines()

# Convert every instance from string to json
embedded_images = [json.loads(image) for image in embedded_images]

# Flatten a list of lists
x = [embedding['embedding'] for embedding in embedded_images]

# Return the image names
y = [embedding['image_id'] for embedding in embedded_images]

def flatten_list(l):
    return [item for sublist in l for item in sublist]

embeddings = [flatten_list(p) for p in x]

# Initialize UMAP with recommended parameters
reducer = umap.UMAP(
    n_neighbors=15,  # Set the number of neighbors to a value greater than 1
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42
)

# Assuming your data is a NumPy array of shape (20000, 65536)
reduced_embeddings = reducer.fit_transform(embeddings)

with open("../../../reduced_embeddings.txt", "w") as f:
    for i in range(len(reduced_embeddings)):
        f.write(json.dumps({"embedding": reduced_embeddings[i].tolist(), "image_id": y[i]}) + "\n")