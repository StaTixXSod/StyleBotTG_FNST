import os
from utils import stylize_image

genre = "abstraction"
number = "1"
content_name = "galgadot"

content_path = f"examples/{content_name}.jpg"
output_path = f"examples/{content_name}_{genre}{number}.jpg"
model_path = f"models/{genre}/{genre}{number}.model"

if os.path.exists(model_path):
    stylize_image(
        content_path=content_path,
        model_path=model_path,
        output_path=output_path,
        device="cpu"
    )
    print(f"Done! Picture has been saved in the: {output_path}")

else:
    print(f"Model {genre}{number} does not exist")