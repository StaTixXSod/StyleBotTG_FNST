from utils import stylize_image

genre = "lights"
number = "2"
content_name = "galgadot"

content_path = f"examples/{content_name}.jpg"
output_path = f"examples/{content_name}_{genre}{number}.jpg"
model_path = f"models/{genre}/{genre}{number}.model"
# output_path = f"examples/{content_name}_vintage_flowers5.jpg"
# model_path = f"models/photo_styles/vintage_flowers5.model"

stylize_image(
    content_path=content_path,
    model_path=model_path,
    output_path=output_path
)
print("Done!")