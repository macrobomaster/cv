from transformers import AutoModelForCausalLM
from PIL import Image

if __name__ == "__main__":
  model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True).to("cuda").eval()
  print("model loaded")

  image = Image.open("test.png")
  print("image loaded")
  enc_image = model.encode_image(image)
  print("image encoded")
  print(model.query(enc_image, "Describe this image."))
