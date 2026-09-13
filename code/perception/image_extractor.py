import base64, json, urllib.request, os
import pandas as pd

images_df = pd.read_csv('dataset/images.csv')
events_df = pd.read_csv('dataset/financial_events.csv')

out_file = 'code/perception/extracted_images.json'
extracted = {}
if os.path.exists(out_file):
    try:
        with open(out_file, 'r', encoding='utf-8') as f:
            extracted = json.load(f)
    except:
        extracted = {}

for idx, row in images_df.iterrows():
    img_id = row['image_id']
    ev_id = row['related_event_id']
    if ev_id in extracted:
        continue
        
    img_path = f"dataset/media/images/{img_id}.png"
    ev_row = events_df[events_df['event_id'] == ev_id].iloc[0]
    
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = (
        f"You are extracting financial values from a document for event '{ev_row['description']}'. "
        f"The currency is {ev_row['currency']}. "
        "Find the final total amount, balance due, net pay, or invoice grand total. "
        "Return strictly JSON: {\"amount\": <number>, \"currency\": \"<currency>\"} with no other text."
    )
    
    payload = {
        'model': 'gemma3:4b',
        'prompt': prompt,
        'images': [img_b64],
        'stream': False,
        'format': 'json'
    }
    
    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode())
            ans = res.get('response')
            data = json.loads(ans)
            amt = float(data.get('amount', 0))
            extracted[ev_id] = {
                'image_id': img_id,
                'event_id': ev_id,
                'amount': amt,
                'currency': data.get('currency', ev_row['currency']),
                'description': ev_row['description']
            }
            print(f"Extracted {img_id} -> {ev_id}: {amt}")
    except Exception as e:
        print(f"Error on {img_id}: {e}")

with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(extracted, f, indent=2, ensure_ascii=False)

print(f"Total extracted: {len(extracted)} saved to {out_file}")
