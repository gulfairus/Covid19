import matplotlib.pyplot as plt
from detection.params import *
from tensorflow import keras
from google.cloud import storage

client = storage.Client()
blobs_scratch = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_outputs/models/cnn_scratch"))
blobs_pre = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_outputs/models/cnn_pre"))
blobs_svm = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_outputs/models/svm"))
blobs_transformer = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_outputs//transformer"))
blobs_xgboost = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="training_outputs/models/xgboost"))

print(len(blobs_scratch))
print(len(blobs_pre))
print(len(blobs_svm))
print(len(blobs_transformer))
print(len(blobs_xgboost))

# def models():
#     try:
#         latest_blob = max(blobs, key=lambda x: x.updated)
#         latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
#         latest_blob.download_to_filename(latest_model_path_to_save)
#         latest_model = keras.models.load_model(latest_model_path_to_save)
#         return latest_model
#     except:
#         return None

# def plot_history(history, title='', axs=None, exp_name=""):
#     if axs is not None:
#         ax1, ax2 = axs
#     else:
#         f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     if len(exp_name) > 0 and exp_name[0] != '_':
#         exp_name = '_' + exp_name
#     ax1.plot(history.history['loss'], label='train' + exp_name)
#     ax1.plot(history.history['val_loss'], label='val' + exp_name)
#     #ax1.set_ylim(0., 2.2)
#     ax1.set_title('loss')
#     ax1.legend()

#     ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
#     ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
#     #ax2.set_ylim(0.25, 1.)
#     ax2.set_title('Accuracy')
#     ax2.legend()
#     return (ax1, ax2)
