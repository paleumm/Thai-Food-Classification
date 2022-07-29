from PIL import Image
from torchvision import transforms
import torch
import streamlit as st
from pytorch_pretrained_vit import ViT
import torch.optim as optim

st.title("Local-Food Image Classification")
st.write("")

image_up = st.file_uploader("Upload food image. (jpg webp) ", type=["jpeg","jpg","webp"])

PATH = 'model_pretrained_True.pth'
IMAGE_SIZE = 224
NUM_CLASSES = 16

model = ViT('B_32_imagenet1k',
    dim=128,
    image_size=224,
    num_classes=NUM_CLASSES
)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# load checkpoint
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

def predict(image):

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    img = Image.open(image)
    batch = torch.unsqueeze(transform(img),0)

    # predict
    model.eval()
    out = model(batch)

    with open('class.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


if image_up is not None:
    # display image that user uploaded
    image = Image.open(image_up).convert('RGB')
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(image_up)

    # predicted class
    for i in labels:
        name = i[0].split()
        st.write("Prediction ", name[1], ",   Score: ", i[1])
    # st.write("Prediction (index, name)", labels[0][0], ",   Score: ", labels[0][1])

st.header("Classes")
st.dataframe(["ส้มตำ", "แกงหน่อไม้", "แกงอ่อม", "ข้าวยำ", "ต้มอึ่ง", "ข้าวซอย", "ผัดไทย", "ข้าวเหนียวมะม่วง", "ข้าวต้มมัด", "แกงขี้เหล็ก", "น้ำพริกอ่อง", "แกงฮังเล", "น้ำพริกหนุ่ม", "ข้าวมันไก่", "แกงไตปลา", "แกงเห็ดเผาะ"],600)