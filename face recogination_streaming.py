from facenet_pytorch import MTCNN, InceptionResnetV1
from FERModel import *
import torch
import numpy as np
import cv2
import os

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
font = cv2.FONT_HERSHEY_TRIPLEX
names = torch.load("./database/names.pt")
embeddings = torch.load("./database/database.pt").to('cuda')

model = FERModel(1, 7)
softmax = torch.nn.Softmax(dim=1)
model.load_state_dict(torch.load('FER2013-Resnet9.pth', map_location=get_default_device()))

def img2tensor(x):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return transform(x)

def predict(img):
    out = model(img2tensor(img)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    return {'label': label, 'probability': prob}

def detect_emotion(img):
    return predict(img)
    
def detect_frame(img):
    faces, boxes = mtcnn(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i, box in enumerate(boxes):
        face_embedding = resnet(faces[i].unsqueeze(0).to('cuda'))
        probs = [(face_embedding - embeddings[i]).norm().item() for i in range(embeddings.size()[0])] 
        name = "Unknow"
        if min(probs) < 0.9:
            index = probs.index(min(probs))
            name = names[index]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        try:
            roi_ = gray[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            resized = cv2.resize(roi_, (48, 48))
            prediction = detect_emotion(resized)
            cv2.putText(img, f"{prediction['label']}", (int(box[0]), int(box[1]) ), font, 1, (255,255,255))
        except:
            pass

        cv2.putText(img, name, (int(box[0] + 6), int(box[3] - 6)), font, 1, (255, 255, 255), 1)
        

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        frame_draw = detect_frame(frame)
        cv2.imshow("0.0", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        cv2.imshow("0.0", frame)

cap.release()
cv2.destroyAllWindows()
print('\nDone')