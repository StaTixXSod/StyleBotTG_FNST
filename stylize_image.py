from utils import stylize_image

genre = "abstraction"
number = "1"
content_name = "galgadot"

content_path = f"examples/{content_name}.jpg"
output_path = f"examples/{content_name}_{genre}{number}.jpg"
model_path = f"models/{genre}/{genre}{number}.model"

stylize_image(
    content_path=content_path,
    model_path=model_path,
    output_path=output_path
)
print(f"Done! Picture has been saved in the: {output_path}")