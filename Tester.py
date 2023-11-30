from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model(model_path):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, sentence):
    labels = ['rstPss', 'chatBot', 'accountSettings', 'reviewFeature', 'editFeature', 'shareReferral']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    percentage_probs = [round(prob, 2) for prob in probs]

    return dict(zip(labels, percentage_probs))

# Load the model
model = load_model('best_model.pt')

# Test sentences
sentences = [
    "application identity manager aim integration technical documentation name of company workfusion website http wwwworkfusioncom name of product intelligent automation smart process automation spa chatbot smartcrowd version intelligent automation sunbird release workfusion solution overview smart process automation spa part of workfusion s intelligent automation product empower enterprise operation to digitize",
    "smart process automation spa combine robotic process automation rpa ai powered cognitive automation and workforce analytic to automate high volume business our late release intelligent automation sunbird release support cloud on premise and hybrid support operation system  cento",
    "workfusion intelligent automation product solve common enterprise automation challenge allow automate large reliable but inflexible fragile core application tackle unstructured datum that raise the need for manual work scale back the need for email chat and phone interaction while increase quality of service optimize and increase people s capacity to do high value work business process design and management",
    "robotic process automation rpa enable rule base automation which eliminate the manual work of operate the user interface of enterprise application like sap and oracle and move structured datum from one system to another",
    "task within business process be automatically route to the right human or bot each action be quality control to ensure accuracy and workload be balanced to ensure optimal capacity utilization",
    "robot username and password workfusion s platform robotic process automation rpa capability be use to automate the operation work in numerous customer legacy and rd party application"
]

# Predict the threat category for each sentence
for sentence in sentences:
    print(f'"{sentence}" \n - Threat predictions: {predict(model, sentence)}\n')
