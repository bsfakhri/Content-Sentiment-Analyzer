import aivm_client as aic

print("Currently supported models:", aic.get_supported_models())

MODEL_NAME = "BertTinySentimentTwitter"
print(f"\nUploading model {MODEL_NAME}...")

try:
    aic.upload_bert_tiny_model("./twitter_bert_tiny.onnx", MODEL_NAME)
    print("Model uploaded successfully!")
    print("\nUpdated supported models:", aic.get_supported_models())
except Exception as e:
    print(f"Error uploading model: {str(e)}")