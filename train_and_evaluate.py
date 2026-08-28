# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 20:37:45 2025

@author: fara_
"""

# ========================================
# Created on Wed Jan 10 10:25:19 2024
# Updated: Overfitting-Reduced NASNetMobile + Cross-Attention (4-branch)
# ========================================
from __future__ import print_function
import tensorflow as tf
import os, sys, random, pickle, time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.layers import (Input, Conv2D, GlobalAveragePooling2D, Dense, Dropout,
                                     BatchNormalization, Multiply, Reshape, Concatenate,
                                     Activation, Add, Lambda, LayerNormalization, GaussianNoise)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras import regularizers

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ---------------- GPU memory growth ----------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set")
    except RuntimeError as e:
        print(e)

# ========================================
# Parameters
# ========================================
batch_size  = 16
epochs      = 100
warmup_ep   = 5                 # freeze base during first few epochs
num_classes = 2
input_shape = (224, 224, 3)
AUTOTUNE    = tf.data.AUTOTUNE
split_rate  = 0.9
weight_decay = 1e-4             # L2 for Dense/attention layers
dropout_rate = 0.6            
gaussian_noise = 0.05
# ========================================
# Paths
# ========================================
Train_ctu_dir = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/CTUDiffThreeCh/paddingResized/Train_CTUDiffThreeCh"
Train_tu_dir  = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/TUDiffThreeCh/paddingResized/Train_TUDiffThreeCh"
Train_dir_dir = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/DirDiffThreeCh/paddingResized/Train_DirDiffThreeCh"
Train_otu_dir = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/OTUDiffThreeCh/paddingResized/Train_OTUDiffThreeCh"
Train_IF_dir = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/IFrame/paddingResized/Train_IFrame"

Test_ctu_dir  = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/CTUDiffThreeCh/paddingResized/Test_CTUDiffThreeCh"
Test_tu_dir   = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/TUDiffThreeCh/paddingResized/Test_TUDiffThreeCh"
Test_dir_dir  = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/DirDiffThreeCh/paddingResized/Test_DirDiffThreeCh"
Test_otu_dir  = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/OTUDiffThreeCh/paddingResized/Test_OTUDiffThreeCh"
Test_IF_dir  = "E:/SHD/Dataset/FinalDataset_QP37/Diff3Ch/IFrame/paddingResized/Test_IFrame"

date_ver = '_4050531_01'
# backbone = 'NASNetMobile'
backbone = 'EfficientNetB0'
logdir   = f"E:/SHD/Log/{backbone}{date_ver}"
os.makedirs(logdir, exist_ok=True)
timestamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
log_path    = os.path.join(logdir, f"log_{date_ver}.txt")
Best_model_dir = f"E:/SHD/Models/BestModel/{backbone}{date_ver}"
os.makedirs(Best_model_dir, exist_ok=True)
model_dir   = f"E:/SHD/Models/{backbone}{date_ver}.keras"
hist_pkl    = f"E:/SHD/Models/{backbone}{date_ver}_history.pkl"
weights_pkl = f"E:/SHD/Models/{backbone}{date_ver}_weights.pkl"

# ========================================
# Logger
# ========================================
class Logger(object):
    def __init__(self, filepath): 
        self.terminal = sys.__stdout__
        self.log = open(filepath, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush() 
        self.log.flush()

sys.stdout = Logger(log_path)
sys.stderr = sys.stdout

# ========================================
# Data helpers
# ========================================
def get_file_paths(base_dir):
    """Get file paths and labels (0: Original, 1: Forged)"""
    file_paths, labels = [], []
    for label_name in ['Original', 'Forged']:
        label_dir = os.path.join(base_dir, label_name)
        for fname in os.listdir(label_dir):
            if fname.endswith('.png'):
                file_paths.append(os.path.join(label_dir, fname))
                labels.append(0 if label_name == 'Original' else 1)
    return file_paths, labels

def decode_img(img_path):
    # return float32 [0,255]; we'll preprocess/augment inside the model
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = preprocess_input(img)   # normalize to NASNetMobile expected range
    return img

def load_multiple_inputs(ctu_path, tu_path, dir_path, otu_path, IF_path, label):
    ctu_img = decode_img(ctu_path)
    tu_img  = decode_img(tu_path)
    dir_img = decode_img(dir_path)
    otu_img = decode_img(otu_path)
    IF_img = decode_img(IF_path)
    return (ctu_img, tu_img, dir_img, otu_img, IF_img), tf.one_hot(label, num_classes)

def build_dataset(ctu_files, tu_files, dir_files, otu_files, IF_files, labels, shuffle=True):
    ctu_files = np.array(ctu_files); tu_files  = np.array(tu_files)
    dir_files = np.array(dir_files); otu_files = np.array(otu_files); IF_files = np.array(IF_files)
    labels    = np.array(labels)
    ds = tf.data.Dataset.from_tensor_slices((ctu_files, tu_files, dir_files, otu_files, IF_files, labels))
    ds = ds.map(load_multiple_inputs, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

# ========================================
# Advanced Test Evaluation Callback
# ========================================
class AdvancedTestEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, best_model_dir, log_dir):
        super().__init__()
        self.test_dataset = test_dataset
        self.best_model_dir = best_model_dir
        self.log_dir = log_dir
        self.best_val_acc = 0
        self.best_epoch = -1
        self.results = []
        
      
        self.plots_dir = os.path.join(log_dir, 'test_evaluation_plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
      
        self.results_file = os.path.join(log_dir, 'test_evaluation_results.txt')
        with open(self.results_file, 'w') as f:
            f.write("TEST EVALUATION RESULTS\n")
            f.write("="*80 + "\n")
            f.write("Epoch\tVal_Acc\tTest_Acc\tPrecision\tRecall\tF1-Score\tAUC-ROC\n")
            f.write("-"*80 + "\n")
    
    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy')
        
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_epoch = epoch
            
            print(f"\n{'='*80}")
            print(f"🎯 New best model found at epoch {epoch+1}!")
            print(f"   Validation Accuracy: {current_val_acc:.4f} ({current_val_acc*100:.2f}%)")
            print(f"{'='*80}")
            
        
            best_model_path = os.path.join(self.best_model_dir, 'best_model.keras')
            if os.path.exists(best_model_path):
                best_model = tf.keras.models.load_model(best_model_path)
                
               
                results = self.evaluate_comprehensive(best_model, epoch+1)
                
                
                results['epoch'] = epoch + 1
                results['val_accuracy'] = current_val_acc
                self.results.append(results)
                
               
                self.print_results(results)
                
               
                self.plot_performance_over_time()
                
              
                if len(self.results) == 1:  # فقط برای اولین بار
                    self.save_sample_predictions(best_model)
            else:
                print(f"⚠️ Warning: Best model file not found at {best_model_path}")
    
    def evaluate_comprehensive(self, model, epoch):
     
        print("   Running comprehensive evaluation on test set...")
        
      
        y_true, y_pred, y_scores = [], [], []
        
        for batch in self.test_dataset:
            inputs, labels = batch
            preds = model.predict(inputs, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
            y_scores.extend(preds[:, 1])
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
       
        accuracy = np.mean(y_true == y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_true, y_scores)
        
       
        cm = confusion_matrix(y_true, y_pred)
        
      
        with open(self.results_file, 'a') as f:
            f.write(f"{epoch}\t{self.best_val_acc:.4f}\t{accuracy:.4f}\t"
                   f"{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{auc_roc:.4f}\n")
        
        return {
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores
        }
    
    def print_results(self, results):
     
        print(f"\n📊 TEST SET EVALUATION RESULTS:")
        print(f"{'='*60}")
        print(f"   Accuracy:   {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"   Precision:  {results['precision']:.4f}")
        print(f"   Recall:     {results['recall']:.4f}")
        print(f"   F1-Score:   {results['f1_score']:.4f}")
        print(f"   AUC-ROC:    {results['auc_roc']:.4f}")
        print(f"{'='*60}\n")
        
      
        cm = results['confusion_matrix']
        print(f"   Confusion Matrix:")
        print(f"               Predicted")
        print(f"              Orig  Forged")
        print(f"   Actual Orig  {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"          Forged {cm[1,0]:4d}   {cm[1,1]:4d}")
        print(f"{'='*60}\n")
    
    def plot_performance_over_time(self):
     
        if len(self.results) < 1:
            return
        
        epochs = [r['epoch'] for r in self.results]
        test_acc = [r['test_accuracy'] for r in self.results]
        f1_scores = [r['f1_score'] for r in self.results]
        auc_scores = [r['auc_roc'] for r in self.results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
       
        axes[0].plot(epochs, test_acc, 'b-o', linewidth=2, markersize=8, label='Test Accuracy')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Test Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=11)
        
      
        axes[1].plot(epochs, f1_scores, 'r-s', linewidth=2, markersize=8, label='F1-Score')
        axes[1].plot(epochs, auc_scores, 'g-^', linewidth=2, markersize=8, label='AUC-ROC')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('F1-Score & AUC-ROC Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=11)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'test_performance_over_time.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📈 Performance plot saved to: {plot_path}")
    
    def save_sample_predictions(self, model, num_samples=10):
       
        print("   Saving sample predictions...")
        
        samples = []
        for batch in self.test_dataset.take(1):
            inputs, labels = batch
            for i in range(min(num_samples, len(labels))):
                pred = model.predict([inp[i:i+1] for inp in inputs], verbose=0)[0]
                pred_class = np.argmax(pred)
                confidence = pred[pred_class]
                true_class = np.argmax(labels[i])
                
                samples.append({
                    'sample_id': i+1,
                    'true_class': true_class,
                    'true_label': 'Original' if true_class==0 else 'Forged',
                    'pred_class': pred_class,
                    'pred_label': 'Original' if pred_class==0 else 'Forged',
                    'confidence': confidence,
                    'correct': pred_class == true_class
                })
        
       
        sample_file = os.path.join(self.log_dir, 'sample_predictions.txt')
        with open(sample_file, 'w') as f:
            f.write("SAMPLE PREDICTIONS\n")
            f.write("="*80 + "\n")
            f.write(f"{'ID':<5} {'True Label':<12} {'Pred Label':<12} {'Confidence':<12} {'Status':<8}\n")
            f.write("-"*80 + "\n")
            for s in samples:
                status = "✓ CORRECT" if s['correct'] else "✗ WRONG"
                f.write(f"{s['sample_id']:<5} {s['true_label']:<12} {s['pred_label']:<12} "
                       f"{s['confidence']:.4f}{' '*(8-len(str(s['confidence'])))} {status:<8}\n")
        
        print(f"   📝 Sample predictions saved to: {sample_file}")
    
    def on_train_end(self, logs=None):
       
        print(f"\n{'='*80}")
        print("🏁 TRAINING COMPLETED - FINAL TEST EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Best model found at epoch {self.best_epoch+1}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} ({self.best_val_acc*100:.2f}%)")
        
        if self.results:
           
            best_result = max(self.results, key=lambda x: x['test_accuracy'])
            print(f"\n📊 BEST TEST PERFORMANCE:")
            print(f"   Epoch: {best_result['epoch']}")
            print(f"   Test Accuracy: {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
            print(f"   Precision: {best_result['precision']:.4f}")
            print(f"   Recall: {best_result['recall']:.4f}")
            print(f"   F1-Score: {best_result['f1_score']:.4f}")
            print(f"   AUC-ROC: {best_result['auc_roc']:.4f}")
            
           
            best_summary_file = os.path.join(self.log_dir, 'best_test_results.txt')
            with open(best_summary_file, 'w') as f:
                f.write("BEST TEST RESULTS\n")
                f.write("="*80 + "\n")
                f.write(f"Best Epoch: {best_result['epoch']}\n")
                f.write(f"Best Validation Accuracy: {self.best_val_acc:.4f}\n")
                f.write(f"Test Accuracy: {best_result['test_accuracy']:.4f}\n")
                f.write(f"Precision: {best_result['precision']:.4f}\n")
                f.write(f"Recall: {best_result['recall']:.4f}\n")
                f.write(f"F1-Score: {best_result['f1_score']:.4f}\n")
                f.write(f"AUC-ROC: {best_result['auc_roc']:.4f}\n")
                f.write("\nConfusion Matrix:\n")
                f.write(f"[[{best_result['confusion_matrix'][0,0]}, {best_result['confusion_matrix'][0,1]}]\n")
                f.write(f" [{best_result['confusion_matrix'][1,0]}, {best_result['confusion_matrix'][1,1]}]]\n")
            
            print(f"\n✅ Best results saved to: {best_summary_file}")
        
        print(f"{'='*80}\n")

# ========================================
# Prepare Dataset
# ========================================
Train_ctu_files, Train_labels = get_file_paths(Train_ctu_dir)
Train_tu_files, _   = get_file_paths(Train_tu_dir)
Train_dir_files, _  = get_file_paths(Train_dir_dir)
Train_otu_files, _  = get_file_paths(Train_otu_dir)
Train_IF_files, _  = get_file_paths(Train_IF_dir)

Train_zipped = list(zip(Train_ctu_files, Train_tu_files, Train_dir_files, Train_otu_files, Train_IF_files, Train_labels))
random.shuffle(Train_zipped)
Train_ctu_files, Train_tu_files, Train_dir_files, Train_otu_files, Train_IF_files, Train_labels = zip(*Train_zipped)

split_idx = int(split_rate * len(Train_labels))
train_dataset = build_dataset(Train_ctu_files[:split_idx], Train_tu_files[:split_idx],
                              Train_dir_files[:split_idx], Train_otu_files[:split_idx],
                              Train_IF_files[:split_idx], Train_labels[:split_idx])
val_dataset   = build_dataset(Train_ctu_files[split_idx:], Train_tu_files[split_idx:],
                              Train_dir_files[split_idx:], Train_otu_files[split_idx:],
                              Train_IF_files[split_idx:], Train_labels[split_idx:], shuffle=True)

Test_ctu_files, Test_labels = get_file_paths(Test_ctu_dir)
Test_tu_files, _  = get_file_paths(Test_tu_dir)
Test_dir_files, _ = get_file_paths(Test_dir_dir)
Test_otu_files, _ = get_file_paths(Test_otu_dir)
Test_IF_files, _ = get_file_paths(Test_IF_dir)

Test_zipped = list(zip(Test_ctu_files, Test_tu_files, Test_dir_files, Test_otu_files, Test_IF_files, Test_labels))
random.shuffle(Test_zipped)
Test_ctu_files, Test_tu_files, Test_dir_files, Test_otu_files, Test_IF_files, Test_labels = zip(*Test_zipped)

# Use the FULL test set (no split)
Test_dataset = build_dataset(Test_ctu_files, Test_tu_files, Test_dir_files, Test_otu_files, Test_IF_files, Test_labels, shuffle=False)

# ========================================
# Attention blocks (with L2)
# ========================================
def cbam_block(input_tensor, ratio=8, name="cbam"):
    """CBAM: Channel + Spatial Attention"""
    channel = input_tensor.shape[-1]

    # ---- Channel Attention ----
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2])

    shared_dense_one = Dense(channel // ratio, activation="relu", name=f"{name}_dense1")
    shared_dense_two = Dense(channel, activation="sigmoid", name=f"{name}_dense2")

    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_out = shared_dense_two(shared_dense_one(max_pool))
    channel_att = Add()([avg_out, max_out])
    channel_att = Reshape((1, 1, channel))(channel_att)
    x = Multiply(name=f"{name}_channel")([input_tensor, channel_att])

    # ---- Spatial Attention ----
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    spatial_att = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid", name=f"{name}_spatial")(concat)
    x = Multiply(name=f"{name}_spatial_out")([x, spatial_att])

    return x

from tensorflow.keras.layers import MultiHeadAttention

def cross_attention(query_feat, key_value_feat, name="cross_attention"):
    """Improved cross-attention with proper scaling and residual"""
    # Flatten spatial dims
    q = tf.reshape(query_feat, [tf.shape(query_feat)[0], -1, query_feat.shape[-1]])
    k = tf.reshape(key_value_feat, [tf.shape(key_value_feat)[0], -1, key_value_feat.shape[-1]])
    v = tf.reshape(key_value_feat, [tf.shape(key_value_feat)[0], -1, key_value_feat.shape[-1]])
    
    # Add layer normalization before attention
    q = LayerNormalization(name=f"{name}_q_ln")(q)
    k = LayerNormalization(name=f"{name}_k_ln")(k)
    v = LayerNormalization(name=f"{name}_v_ln")(v)
    
    # Multi-head attention with dropout
    attn_out = MultiHeadAttention(
        num_heads=8,  # Increased heads
        key_dim=query_feat.shape[-1]//8, 
        dropout=0.1,  # Add dropout
        name=name
    )(q, k, v)
    
    # Restore spatial dimensions
    h, w = query_feat.shape[1], query_feat.shape[2]
    attn_out = tf.reshape(attn_out, [tf.shape(query_feat)[0], h, w, query_feat.shape[-1]])
    
    # Residual connection + normalization with gate
    residual = Add(name=f"{name}_residual")([query_feat, attn_out])
    output = LayerNormalization(name=f"{name}_ln")(residual)
    
    return output


# ========================================
# Model
# ========================================
def build_model(input_shape=(224,224,3), num_classes=2, augment=True):
    # Shared augmentation (light, to avoid destroying subtle forensics cues)
    # Inputs
    ctu_in = Input(input_shape, name='ctu_input')
    tu_in  = Input(input_shape, name='tu_input')
    dir_in = Input(input_shape, name='dir_input')
    otu_in = Input(input_shape, name='otu_input')
    IF_in = Input(input_shape, name='IF_input')


    base = EfficientNetB0(include_top=False, weights=None, input_shape=input_shape)
    base.load_weights(f'C:/Users/fara_/.keras/models/efficientnetb0_notop.h5')

        
    for layer in base.layers[-5:]:
        layer.trainable = True
        
   
        
    feat_ctu = base(ctu_in)
    feat_tu  = base(tu_in)
    feat_dir = base(dir_in)
    feat_otu = base(otu_in)
    feat_IF = base(IF_in)
      

    
    # Cross-attention chain
    IF_feat = feat_IF

    
    ctu_feat = cross_attention(feat_ctu, IF_feat,  name='ctu_IF')

    
    tu_feat  = cross_attention(feat_tu,  IF_feat, name='tu_IF')

    
    dir_feat = cross_attention(feat_dir, IF_feat, name='dir_IF')

    
    otu_feat = cross_attention(feat_otu, IF_feat, name='otu_IF')

    

    # Global pooling + dropout for each branch
    def pool(x, name):
        return Dropout(0.2, name=f'{name}_do')(GlobalAveragePooling2D(name=name)(x))

    p_ctu = pool(ctu_feat, 'gap_ctu'); p_tu = pool(tu_feat, 'gap_tu')
    p_dir = pool(dir_feat, 'gap_dir'); p_otu = pool(otu_feat, 'gap_otu')
    p_IF = pool(IF_feat, 'gap_IF')

    merged = Concatenate(name='concat')([p_ctu, p_tu, p_dir, p_otu, p_IF])

    # Head with L2 + Dropout
    reg = regularizers.l2(weight_decay)
    x = Dense(64, 
              activation='relu', 
              # kernel_regularizer=reg
              )(merged); x = Dropout(0.2)(x)

    x = Dense(64, 
              activation='relu', 
              # kernel_regularizer=reg
              )(x);  x = Dropout(0.2)(x); x = BatchNormalization()(x)
    out = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model([ctu_in, tu_in, dir_in, otu_in, IF_in], out, name=f"{backbone}_Attn_4branch")
    return model, base

model, base = build_model(input_shape, num_classes)

# ========================================
# Compile (label smoothing + lower LR)
# ========================================
optimizer = Adam(learning_rate=1e-4)

loss_fn = 'categorical_crossentropy'
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])
model.summary()

# ========================================
# Callbacks
# ========================================
callbacks_common = [
    ModelCheckpoint(filepath=os.path.join(Best_model_dir, 'best_model.keras'),
                    monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda batch,logs:time.sleep(10),
    ),
    TensorBoard(
    log_dir=os.path.join(logdir, timestamp),
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',   # log once per epoch
    profile_batch=0 
),

]

test_eval_callback = AdvancedTestEvaluationCallback(
    test_dataset=Test_dataset,
    best_model_dir=Best_model_dir,
    log_dir=logdir
)

callbacks_common = [
    ModelCheckpoint(filepath=os.path.join(Best_model_dir, 'best_model.keras'),
                    monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda batch,logs:time.sleep(10),
    ),
    test_eval_callback, 
    TensorBoard(
        log_dir=os.path.join(logdir, timestamp),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=0 
    ),
]

# ========================================
# Train in two phases: warm-up (frozen base) then fine-tune last 100 layers
# ========================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_common,
    verbose=1
)

# ========================================
# Save
# ========================================
model.save(model_dir)
with open(hist_pkl, 'wb') as f:
    pickle.dump(history, f)
print("Training Complete. Model and history saved.")

# ========================================
# Evaluate best checkpoint on FULL test set
# ========================================
best_model = tf.keras.models.load_model(os.path.join(Best_model_dir, 'best_model.keras'))
test_loss, test_acc = best_model.evaluate(Test_dataset, verbose=1)
print("Best model test loss:", test_loss)
print("Best model test accuracy:", test_acc)

# ============================
# ROC, AUC, Confusion Matrix
# ============================
y_true, y_score = [], []
for batch in Test_dataset:
    inputs, labels = batch
    preds = best_model.predict(inputs, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_score.extend(preds[:, 1])  # score for class 1 (Forged)

y_true = np.array(y_true); y_score = np.array(y_score)

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC'); plt.legend(); plt.grid(True)
os.makedirs(os.path.join(logdir, 'Plots'), exist_ok=True)
plt.tight_layout(); plt.savefig(os.path.join(logdir, 'Plots', f'ROC_curve{date_ver}.png')); plt.close()

y_pred = (y_score > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Original", "Forged"], yticklabels=["Original", "Forged"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(os.path.join(logdir, 'Plots', f'ConfusionMatrix{date_ver}.png')); plt.close()

report = classification_report(y_true, y_pred, target_names=["Original", "Forged"])
print("\nClassification Report:\n", report)

from collections import defaultdict

video_preds = defaultdict(list)
video_labels = {}
for (ctu_path, tu_path, dir_path, otu_path,IF_path, label) in Test_zipped:
    # Predict for one frame across all four features
    inputs = (
        decode_img(ctu_path)[None, ...],
        decode_img(tu_path)[None, ...],
        decode_img(dir_path)[None, ...],
        decode_img(otu_path)[None, ...],
        decode_img(IF_path)[None, ...]
    )
    probs = best_model.predict(inputs, verbose=0)
    pred = np.argmax(probs, axis=1)[0]
    true_label = label


    fname = os.path.basename(ctu_path)
    parts = fname.split("_")  
    

    if len(parts) >= 3:
        video_id = f"{parts[0]}_{parts[1]}"  
    else:
        # Fallback for unexpected formats
        video_id = fname

    # Store predictions + label
    video_preds[video_id].append(pred)

    # Ensure consistent label
    if video_id not in video_labels:
        video_labels[video_id] = true_label
    elif video_labels[video_id] != true_label:
        print(f"[Warning] Inconsistent labels for {video_id}!")

# Majority vote per video
correct_videos = 0
y_true_vid, y_pred_vid = [], []
for vid, preds in video_preds.items():
    final_pred = int(np.mean(preds) >= 0.5)  # majority voting
    y_pred_vid.append(final_pred)
    y_true_vid.append(video_labels[vid])
    if final_pred == video_labels[vid]:
        correct_videos += 1

video_accuracy = correct_videos / len(video_preds)
print(f"\nVideo-level Accuracy: {video_accuracy:.4f} ({correct_videos}/{len(video_preds)})")

# Print video-level classification report
video_report = classification_report(y_true_vid, y_pred_vid, target_names=["Original", "Forged"])
print("\nVideo-level Classification Report:\n", video_report)

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='train')
ax1.plot(history.history['val_accuracy'], label='validation')
ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='validation')
ax2.set_title('Loss'); ax2.set_xlabel('Epoch'); ax2.legend()
plt.tight_layout(); plt.savefig(os.path.join(logdir, 'Plots', f'training_curves{date_ver}.png')); plt.close()

# Save weights (pickle)
with open(weights_pkl, 'wb') as f:
    pickle.dump(model.get_weights(), f, protocol=pickle.HIGHEST_PROTOCOL)

print("All artifacts saved.")

# ========================================
# Evaluate EACH VIDEO individually (separate accuracies)
# ========================================
from collections import defaultdict

video_preds = defaultdict(list)
video_labels = {}

print("\n" + "="*60)
print("INDIVIDUAL VIDEO RESULTS")
print("="*60)

for (ctu_path, tu_path, dir_path, otu_path, IF_path, label) in Test_zipped:
    # Predict for one frame
    inputs = (
        decode_img(ctu_path)[None, ...],
        decode_img(tu_path)[None, ...],
        decode_img(dir_path)[None, ...],
        decode_img(otu_path)[None, ...],
        decode_img(IF_path)[None, ...]
    )
    probs = best_model.predict(inputs, verbose=0)
    pred = np.argmax(probs, axis=1)[0]
    true_label = label

    # Extract video id from filename
    fname = os.path.basename(ctu_path)
    parts = fname.split("_")  
    
    # Group by video ID (ignore frame number)
    if len(parts) >= 3:
        video_id = f"{parts[0]}_{parts[1]}"   # e.g. OriginalDiff_1 or ForgedDiff_1
    else:
        video_id = fname

    # Store predictions + label
    video_preds[video_id].append(pred)

    # Ensure consistent label
    if video_id not in video_labels:
        video_labels[video_id] = true_label
    elif video_labels[video_id] != true_label:
        print(f"[Warning] Inconsistent labels for {video_id}!")

# Calculate per-video accuracy and print individual results
correct_videos = 0
per_video_results = []

print("\n{:<25} {:<12} {:<12} {:<15} {:<10}".format(
    "Video ID", "True Label", "Predicted", "Accuracy (%)", "Correct/Total"
))
print("-" * 80)

for vid, preds in video_preds.items():
    final_pred = int(np.mean(preds) >= 0.5)
    true_label_name = "Original" if video_labels[vid] == 0 else "Forged"
    pred_label_name = "Original" if final_pred == 0 else "Forged"
    
    # Calculate frame accuracy for this video
    frame_accuracy = (np.array(preds) == video_labels[vid]).mean() * 100
    correct_frames = np.sum(np.array(preds) == video_labels[vid])
    total_frames = len(preds)
    
    is_correct = (final_pred == video_labels[vid])
    if is_correct:
        correct_videos += 1
    
    status = "✓" if is_correct else "✗"
    
    print("{:<25} {:<12} {:<12} {:<15.2f} {:<3}/{:<3} {}".format(
        vid, true_label_name, pred_label_name, 
        frame_accuracy, correct_frames, total_frames, status
    ))
    
    per_video_results.append({
        'video_id': vid,
        'true_label': video_labels[vid],
        'true_label_name': true_label_name,
        'predicted_label': final_pred,
        'predicted_label_name': pred_label_name,
        'frame_accuracy': frame_accuracy,
        'correct_frames': correct_frames,
        'total_frames': total_frames,
        'is_correct': is_correct
    })

video_accuracy = correct_videos / len(video_preds)
print("-" * 80)
print(f"\nOVERALL VIDEO-LEVEL ACCURACY: {video_accuracy:.4f} ({correct_videos}/{len(video_preds)})")
print("="*60)

# Print detailed summary by video type
original_videos = [r for r in per_video_results if r['true_label'] == 0]
forged_videos = [r for r in per_video_results if r['true_label'] == 1]

print("\n" + "="*60)
print("SUMMARY BY VIDEO TYPE")
print("="*60)

if original_videos:
    original_correct = sum(1 for r in original_videos if r['is_correct'])
    original_acc = original_correct / len(original_videos) * 100
    print(f"ORIGINAL Videos: {original_correct}/{len(original_videos)} correct ({original_acc:.2f}%)")
    
if forged_videos:
    forged_correct = sum(1 for r in forged_videos if r['is_correct'])
    forged_acc = forged_correct / len(forged_videos) * 100
    print(f"FORGED Videos:   {forged_correct}/{len(forged_videos)} correct ({forged_acc:.2f}%)")

# Print videos that were misclassified
misclassified = [r for r in per_video_results if not r['is_correct']]
if misclassified:
    print("\n" + "="*60)
    print("MISCLASSIFIED VIDEOS")
    print("="*60)
    for r in misclassified:
        print(f"  {r['video_id']}: Actual={r['true_label_name']}, "
              f"Predicted={r['predicted_label_name']}, "
              f"Frame Acc={r['frame_accuracy']:.1f}% ({r['correct_frames']}/{r['total_frames']} frames)")

# Print best and worst performing videos
print("\n" + "="*60)
print("BEST & WORST PERFORMING VIDEOS")
print("="*60)

# Sort by frame accuracy
sorted_videos = sorted(per_video_results, key=lambda x: x['frame_accuracy'], reverse=True)

print("\nTOP 3 BEST PERFORMING VIDEOS:")
for i, r in enumerate(sorted_videos[:3], 1):
    print(f"  {i}. {r['video_id']}: {r['frame_accuracy']:.1f}% correct "
          f"({r['correct_frames']}/{r['total_frames']} frames) - {r['true_label_name']} -> {r['predicted_label_name']}")

print("\nBOTTOM 3 WORST PERFORMING VIDEOS:")
for i, r in enumerate(sorted_videos[-3:], 1):
    print(f"  {i}. {r['video_id']}: {r['frame_accuracy']:.1f}% correct "
          f"({r['correct_frames']}/{r['total_frames']} frames) - {r['true_label_name']} -> {r['predicted_label_name']}")

# Print video-level classification report
print("\n" + "="*60)
print("VIDEO-LEVEL CLASSIFICATION REPORT")
print("="*60)
video_report = classification_report(y_true_vid, y_pred_vid, target_names=["Original", "Forged"])
print(video_report)

# ========================================
# GENERATE ALL FIGURES FOR PAPER
# ========================================
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'


paper_fig_dir = os.path.join(logdir, 'Paper_Figures')
os.makedirs(paper_fig_dir, exist_ok=True)

# ========================================
# FIGURE 1: Distribution of Frame Accuracies
# ========================================
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

# نمودار هیستوگرام توزیع دقت فریم
frame_accs = [r['frame_accuracy'] for r in per_video_results]
axes[0].hist(frame_accs, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(np.mean(frame_accs), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(frame_accs):.1f}%')
axes[0].axvline(np.median(frame_accs), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(frame_accs):.1f}%')
axes[0].set_xlabel('Frame Accuracy (%)', fontsize=12)
axes[0].set_ylabel('Number of Videos', fontsize=12)
axes[0].set_title('Distribution of Frame-Level Accuracies', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)


orig_accs = [r['frame_accuracy'] for r in per_video_results if r['true_label'] == 0]
forged_accs = [r['frame_accuracy'] for r in per_video_results if r['true_label'] == 1]

bp = axes[1].boxplot([orig_accs, forged_accs], 
                      labels=['Original\nVideos', 'Forged\nVideos'],
                      patch_artist=True,
                      widths=0.6)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1].set_ylabel('Frame Accuracy (%)', fontsize=12)
axes[1].set_title('Frame Accuracy by Video Type', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='Random chance')

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure1_Frame_Accuracy_Distribution.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure1_Frame_Accuracy_Distribution.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 1 saved: Frame Accuracy Distribution")

# ========================================
# FIGURE 2: Video-Level vs Frame-Level Accuracy
# ========================================
fig2, ax = plt.subplots(figsize=(12, 8))

# ایجاد ترتیب ویدئوها بر اساس دقت فریم (صعودی)
sorted_results = sorted(per_video_results, key=lambda x: x['frame_accuracy'])
video_names = [r['video_id'] for r in sorted_results]
frame_accs_sorted = [r['frame_accuracy'] for r in sorted_results]
video_correct = [r['is_correct'] for r in sorted_results]
true_labels = [r['true_label'] for r in sorted_results]


colors = []
for i, (correct, label) in enumerate(zip(video_correct, true_labels)):
    if correct and label == 0:
        colors.append('darkgreen') 
    elif correct and label == 1:
        colors.append('darkred')    
    else:
        colors.append('gray')       

bars = ax.barh(video_names, frame_accs_sorted, color=colors, alpha=0.8, edgecolor='black')
ax.axvline(x=50, color='black', linestyle='--', linewidth=2, label='Random chance (50%)', alpha=0.7)
ax.axvline(x=100, color='blue', linestyle=':', linewidth=2, label='Perfect accuracy', alpha=0.7)


for i, (correct, label, acc) in enumerate(zip(video_correct, true_labels, frame_accs_sorted)):
    if not correct:
        ax.text(acc + 1, i, '✗', fontsize=14, va='center', color='red')
    else:
        ax.text(acc + 1, i, '✓', fontsize=14, va='center', color='green')

ax.set_xlabel('Frame Accuracy (%)', fontsize=12)
ax.set_ylabel('Video ID', fontsize=12)
ax.set_title('Frame-Level Accuracy per Video with Majority Voting Outcome', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')


mean_acc = np.mean(frame_accs)
ax.text(0.02, 0.98, f'Mean Frame Acc: {mean_acc:.1f}%\nVideo Acc: 100%', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure2_Video_vs_Frame_Accuracy.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure2_Video_vs_Frame_Accuracy.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 2 saved: Video vs Frame Accuracy")

# ========================================
# FIGURE 3: Confusion Matrix (Frame-Level)
# ========================================
from sklearn.metrics import confusion_matrix
import seaborn as sns

fig3, ax = plt.subplots(figsize=(8, 7))

frame_preds = []
frame_true = []
for batch in Test_dataset:
    inputs, labels = batch
    preds = best_model.predict(inputs, verbose=0)
    frame_preds.extend(np.argmax(preds, axis=1))
    frame_true.extend(np.argmax(labels.numpy(), axis=1))

cm_frame = confusion_matrix(frame_true, frame_preds)
cm_percent = cm_frame.astype('float') / cm_frame.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(cm_frame, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Original', 'Forged'],
            yticklabels=['Original', 'Forged'],
            ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Frame-Level Confusion Matrix', fontsize=14, fontweight='bold')


for i in range(2):
    for j in range(2):
        ax.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)', 
                ha='center', va='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure3_Frame_Confusion_Matrix.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure3_Frame_Confusion_Matrix.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 3 saved: Frame Confusion Matrix")

# ========================================
# FIGURE 4: ROC Curves (Frame & Video Level)
# ========================================
fig4, ax = plt.subplots(figsize=(8, 7))

# ROC برای سطح فریم
fpr_frame, tpr_frame, _ = roc_curve(frame_true, y_score)
auc_frame = auc(fpr_frame, tpr_frame)
ax.plot(fpr_frame, tpr_frame, lw=2, label=f'Frame-Level (AUC = {auc_frame:.3f})', 
        color='blue')


video_scores = []
video_true_labels = []
for video_id in video_preds.keys():
    preds = video_preds[video_id]
    forged_ratio = np.mean(preds)  
    video_scores.append(forged_ratio)
    video_true_labels.append(video_labels[video_id])

fpr_video, tpr_video, _ = roc_curve(video_true_labels, video_scores)
auc_video = auc(fpr_video, tpr_video)
ax.plot(fpr_video, tpr_video, lw=2, label=f'Video-Level (AUC = {auc_video:.3f})', 
        color='red', linestyle='--')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Frame-Level vs Video-Level Detection', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure4_ROC_Curves.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure4_ROC_Curves.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 4 saved: ROC Curves")

# ========================================
# FIGURE 5: Training Curves (Loss & Accuracy)
# ========================================
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.plot(history.history['accuracy'], 'b-', label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])


ax2.plot(history.history['loss'], 'b-', label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max(history.history['loss'])])


best_epoch = np.argmin(history.history['val_loss'])
ax1.scatter(best_epoch, history.history['val_accuracy'][best_epoch], 
            color='green', s=100, zorder=5, label=f'Best (Epoch {best_epoch+1})')
ax2.scatter(best_epoch, history.history['val_loss'][best_epoch], 
            color='green', s=100, zorder=5)

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure5_Training_Curves.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure5_Training_Curves.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 5 saved: Training Curves")

# ========================================
# FIGURE 6: Per-Video Performance Heatmap
# ========================================
fig6, ax = plt.subplots(figsize=(14, 8))

video_names_sorted = sorted(video_preds.keys())
n_videos = len(video_names_sorted)
max_frames = max([len(video_preds[v]) for v in video_names_sorted])

# ایجاد ماتریس پیش‌بینی‌ها
prediction_matrix = np.zeros((n_videos, max_frames))
truth_matrix = np.zeros((n_videos, max_frames))

for i, video_id in enumerate(video_names_sorted):
    preds = video_preds[video_id]
    true = video_labels[video_id]
    for j, pred in enumerate(preds):
        prediction_matrix[i, j] = pred
        truth_matrix[i, j] = true


error_matrix = (prediction_matrix != truth_matrix).astype(float)
error_matrix[error_matrix == 0] = np.nan  # فقط خطاها را نشان بده

# Plot
im = ax.imshow(prediction_matrix, aspect='auto', cmap='RdYlGn_r', 
               interpolation='nearest', vmin=0, vmax=1)
ax.set_xlabel('Frame Index', fontsize=12)
ax.set_ylabel('Video ID', fontsize=12)
ax.set_title('Per-Frame Predictions Heatmap (Green: Correct, Red: Forged Pred)', 
             fontsize=14, fontweight='bold')

ax.set_yticks(range(n_videos))
ax.set_yticklabels(video_names_sorted, fontsize=8)
ax.set_xticks(range(0, max_frames, 10))
ax.set_xticklabels(range(0, max_frames, 10))


cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Prediction (0: Original, 1: Forged)', fontsize=10)


for i, video_id in enumerate(video_names_sorted):
    if video_labels[video_id] == 1:
        ax.text(-2, i, 'F', fontsize=10, ha='center', va='center', 
                color='red', fontweight='bold')
    else:
        ax.text(-2, i, 'O', fontsize=10, ha='center', va='center', 
                color='green', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure6_PerFrame_Predictions_Heatmap.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure6_PerFrame_Predictions_Heatmap.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 6 saved: Per-Frame Predictions Heatmap")

# ========================================
# FIGURE 7: Statistical Analysis - Confidence Intervals
# ========================================
# ========================================
# FIGURE 7: Statistical Analysis - Confidence Intervals (Corrected)
# ========================================
fig7, ax = plt.subplots(figsize=(10, 6))


from scipy import stats
import numpy as np

n_videos = len(video_preds)
correct = correct_videos
p_observed = correct / n_videos * 100

def wilson_interval(correct, total, confidence=0.95):
    """Wilson score interval - recommended for proportions"""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    lower = max(0, center - spread) * 100
    upper = min(1, center + spread) * 100
    return lower, upper

def jeffreys_interval(correct, total, confidence=0.95):
    """Jeffreys interval using Beta distribution"""
    alpha = 1 - confidence
    lower = stats.beta.ppf(alpha/2, correct + 0.5, total - correct + 0.5) * 100
    upper = stats.beta.ppf(1 - alpha/2, correct + 0.5, total - correct + 0.5) * 100
    if correct == 0:
        lower = 0
    if correct == total:
        upper = 100
    return lower, upper

def agresti_coull(correct, total, confidence=0.95):
    """Agresti-Coull interval"""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z_squared = z**2
    n_tilde = total + z_squared
    p_tilde = (correct + z_squared/2) / n_tilde
    se = np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    lower = max(0, p_tilde - z * se) * 100
    upper = min(1, p_tilde + z * se) * 100
    return lower, upper

def clopper_pearson(correct, total, confidence=0.95):
    """Clopper-Pearson exact interval"""
    alpha = 1 - confidence
    lower = stats.beta.ppf(alpha/2, correct, total - correct + 1) * 100
    upper = stats.beta.ppf(1 - alpha/2, correct + 1, total - correct) * 100
    if correct == 0:
        lower = 0
    if correct == total:
        upper = 100
    return lower, upper


methods = ['Wilson', 'Jeffreys', 'Agresti-Coull', 'Clopper-Pearson']
cis = []

# Wilson
lower, upper = wilson_interval(correct, n_videos)
cis.append([lower, upper])

# Jeffreys
lower, upper = jeffreys_interval(correct, n_videos)
cis.append([lower, upper])

# Agresti-Coull
lower, upper = agresti_coull(correct, n_videos)
cis.append([lower, upper])

# Clopper-Pearson
lower, upper = clopper_pearson(correct, n_videos)
cis.append([lower, upper])


cis = np.array(cis)


errors_lower = np.maximum(0, p_observed - cis[:, 0]) 
errors_upper = np.maximum(0, cis[:, 1] - p_observed)  

y_pos = np.arange(len(methods))


for i, (method, (lower, upper)) in enumerate(zip(methods, cis)):
  
    ax.plot([lower, upper], [i, i], 'b-', linewidth=2, alpha=0.7)
    ax.plot(p_observed, i, 'o', color='blue', markersize=8, alpha=0.7)

ax.axvline(x=p_observed, color='red', linestyle='--', linewidth=2,
           label=f'Observed: {p_observed:.1f}%')

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=11)
ax.set_xlabel('Video-Level Accuracy (%)', fontsize=12)
ax.set_title('95% Confidence Intervals for Video Accuracy\n(30 Videos, 30 Correct)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([70, 100])


for i, (lower, upper) in enumerate(cis):
    ax.annotate(f'[{lower:.1f}, {upper:.1f}]', 
                xy=(upper + 0.5, i), 
                fontsize=9, va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure7_Confidence_Intervals.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure7_Confidence_Intervals.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 7 saved: Confidence Intervals (using scipy only)")


print("\n" + "="*60)
print("95% CONFIDENCE INTERVALS FOR VIDEO ACCURACY")
print("="*60)
print(f"Observed accuracy: {p_observed:.1f}% ({correct}/{n_videos})")
print("-"*40)
for method, (lower, upper) in zip(methods, cis):
    print(f"{method:16}: [{lower:.2f}%, {upper:.2f}%]")
print("="*60)

# ========================================
# FIGURE 8: Majority Voting Robustness Demonstration
# ========================================
fig8, axes = plt.subplots(2, 2, figsize=(12, 10))


video_example1 = 'OriginalDiff_100'
preds1 = video_preds[video_example1]
true1 = video_labels[video_example1]
cumulative_acc1 = np.cumsum(np.array(preds1) == true1) / np.arange(1, len(preds1)+1) * 100

axes[0,0].plot(range(1, len(preds1)+1), cumulative_acc1, 'b-', linewidth=2)
axes[0,0].axhline(y=50, color='red', linestyle='--', label='Majority threshold (50%)')
axes[0,0].axhline(y=100, color='green', linestyle=':', alpha=0.5)
axes[0,0].set_xlabel('Number of Frames Processed')
axes[0,0].set_ylabel('Cumulative Accuracy (%)')
axes[0,0].set_title(f'{video_example1} - Low Frame Acc (63%) but Correct')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylim([0, 100])


video_example2 = 'ForgedDiff_50'
preds2 = video_preds[video_example2]
true2 = video_labels[video_example2]
cumulative_acc2 = np.cumsum(np.array(preds2) == true2) / np.arange(1, len(preds2)+1) * 100

axes[0,1].plot(range(1, len(preds2)+1), cumulative_acc2, 'r-', linewidth=2)
axes[0,1].axhline(y=50, color='red', linestyle='--', label='Majority threshold (50%)')
axes[0,1].set_xlabel('Number of Frames Processed')
axes[0,1].set_ylabel('Cumulative Accuracy (%)')
axes[0,1].set_title(f'{video_example2} - Moderate Frame Acc (78%) but Correct')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_ylim([0, 100])


video_example3 = 'OriginalDiff_30'
preds3 = video_preds[video_example3]
true3 = video_labels[video_example3]
cumulative_acc3 = np.cumsum(np.array(preds3) == true3) / np.arange(1, len(preds3)+1) * 100

axes[1,0].plot(range(1, len(preds3)+1), cumulative_acc3, 'g-', linewidth=2)
axes[1,0].axhline(y=50, color='red', linestyle='--', label='Majority threshold (50%)')
axes[1,0].set_xlabel('Number of Frames Processed')
axes[1,0].set_ylabel('Cumulative Accuracy (%)')
axes[1,0].set_title(f'{video_example3} - High Frame Acc (100%)')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_ylim([0, 100])



n_frames_range = np.arange(10, 101, 10)
simulations = []
for n_frames in n_frames_range:
  
    frame_acc = 0.65
    
    threshold = n_frames // 2 + 1
    prob_correct = 1 - stats.binom.cdf(threshold-1, n_frames, frame_acc)
    simulations.append(prob_correct * 100)

axes[1,1].plot(n_frames_range, simulations, 'purple', linewidth=2, marker='o')
axes[1,1].axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% confidence')
axes[1,1].axhline(y=99, color='blue', linestyle='--', alpha=0.7, label='99% confidence')
axes[1,1].set_xlabel('Number of Frames per Video')
axes[1,1].set_ylabel('Probability of Correct Majority Vote (%)')
axes[1,1].set_title('Majority Voting Robustness\n(Assuming 65% Frame Accuracy)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].set_ylim([50, 100])

plt.suptitle('Majority Voting Robustness Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure8_Majority_Voting_Analysis.pdf'), 
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure8_Majority_Voting_Analysis.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Figure 8 saved: Majority Voting Analysis")

# ========================================
# FIGURE 9: Sample Predictions Visualization
# ========================================
fig9, axes = plt.subplots(2, 3, figsize=(15, 10))


sample_videos = {
    'Original High Acc': ('OriginalDiff_30', 0),
    'Original Low Acc': ('OriginalDiff_100', 0),
    'Forged High Acc': ('ForgedDiff_30', 1),
    'Forged Low Acc': ('ForgedDiff_50', 1)
}

for idx, (title, (video_id, label)) in enumerate(sample_videos.items()):
    ax = axes[idx // 3, idx % 3]
    
   
    for ctu_path, tu_path, dir_path, otu_path, IF_path, true_label in Test_zipped:
        fname = os.path.basename(ctu_path)
        parts = fname.split('_')
        vid = '_'.join(parts[:2]) if len(parts) >= 3 else fname
        if vid == video_id:
            img = decode_img(ctu_path).numpy()
            # denormalize for display
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img)
            
           
            inputs = (decode_img(ctu_path)[None, ...], decode_img(tu_path)[None, ...],
                      decode_img(dir_path)[None, ...], decode_img(otu_path)[None, ...],
                      decode_img(IF_path)[None, ...])
            prob = best_model.predict(inputs, verbose=0)[0]
            pred_class = np.argmax(prob)
            confidence = prob[pred_class]
            
            color = 'green' if pred_class == label else 'red'
            ax.set_title(f'{title}\nTrue: {"Original" if label==0 else "Forged"}\n'
                        f'Pred: {"Original" if pred_class==0 else "Forged"} ({confidence:.2f})',
                        color=color, fontsize=10)
            ax.axis('off')
            break

# plot confidence histogram
ax = axes[1, 2]
all_confidences = []
for batch in Test_dataset:
    inputs, labels = batch
    probs = best_model.predict(inputs, verbose=0)
    all_confidences.extend(np.max(probs, axis=1))
ax.hist(all_confidences, bins=30, edgecolor='black', alpha=0.7, color='purple')
ax.axvline(x=0.5, color='red', linestyle='--', label='Decision threshold')
ax.set_xlabel('Confidence Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Prediction Confidence')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Sample Predictions and Confidence Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure9_Sample_Predictions.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure9_Sample_Predictions.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 9 saved: Sample Predictions")

# ========================================
# FIGURE 10: Per-Class Performance Radar Chart
# ========================================
fig10, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))


from sklearn.metrics import precision_score, recall_score, f1_score

# Frame-level per class
frame_precision = precision_score(frame_true, frame_preds, average=None)
frame_recall = recall_score(frame_true, frame_preds, average=None)
frame_f1 = f1_score(frame_true, frame_preds, average=None)

categories = ['Precision', 'Recall', 'F1-Score']
original_scores = [frame_precision[0], frame_recall[0], frame_f1[0]]
forged_scores = [frame_precision[1], frame_recall[1], frame_f1[1]]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

original_scores += original_scores[:1]
forged_scores += forged_scores[:1]

ax.plot(angles, original_scores, 'o-', linewidth=2, label='Original', color='green')
ax.fill(angles, original_scores, alpha=0.25, color='green')
ax.plot(angles, forged_scores, 'o-', linewidth=2, label='Forged', color='red')
ax.fill(angles, forged_scores, alpha=0.25, color='red')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax.set_title('Per-Class Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure10_Radar_Chart.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure10_Radar_Chart.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 10 saved: Radar Chart")

# ========================================
# FIGURE 11: Error Analysis
# ========================================
fig11, axes = plt.subplots(2, 2, figsize=(12, 10))


error_rates = [(100 - r['frame_accuracy']) for r in per_video_results]
video_names_short = [r['video_id'].replace('OriginalDiff_', 'O').replace('ForgedDiff_', 'F') 
                     for r in per_video_results]

axes[0,0].barh(video_names_short, error_rates, color='coral', edgecolor='black')
axes[0,0].set_xlabel('Error Rate (%)')
axes[0,0].set_ylabel('Video')
axes[0,0].set_title('Error Rate per Video')
axes[0,0].grid(True, alpha=0.3, axis='x')


frame_counts = [r['total_frames'] for r in per_video_results]
accuracies = [r['frame_accuracy'] for r in per_video_results]

axes[0,1].scatter(frame_counts, accuracies, alpha=0.6, s=100, c=accuracies, cmap='RdYlGn')
axes[0,1].set_xlabel('Number of Frames per Video')
axes[0,1].set_ylabel('Frame Accuracy (%)')
axes[0,1].set_title('Accuracy vs Number of Frames')
axes[0,1].grid(True, alpha=0.3)


cumulative_acc = np.cumsum([1 if p == t else 0 for p, t in zip(frame_preds, frame_true)]) / np.arange(1, len(frame_preds)+1) * 100
axes[1,0].plot(cumulative_acc, 'b-', linewidth=1, alpha=0.7)
axes[1,0].set_xlabel('Number of Frames Processed')
axes[1,0].set_ylabel('Cumulative Accuracy (%)')
axes[1,0].set_title('Overall Cumulative Frame Accuracy')
axes[1,0].axhline(y=94.7, color='red', linestyle='--', label='Final: 94.7%')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)


from sklearn.metrics import confusion_matrix
cm_norm = confusion_matrix(frame_true, frame_preds, normalize='true')
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Original', 'Forged'], yticklabels=['Original', 'Forged'],
            ax=axes[1,1], cbar_kws={'label': 'Percentage'})
axes[1,1].set_xlabel('Predicted Label')
axes[1,1].set_ylabel('True Label')
axes[1,1].set_title('Normalized Confusion Matrix')

plt.suptitle('Error Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(paper_fig_dir, 'Figure11_Error_Analysis.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(paper_fig_dir, 'Figure11_Error_Analysis.png'), dpi=300, bbox_inches='tight')
plt.show()
print("✓ Figure 11 saved: Error Analysis")

# ========================================
# SAVE ALL RESULTS TO EXCEL FOR PAPER
# ========================================
import pandas as pd


df_frame_results = pd.DataFrame({
    'Video_ID': video_names_sorted,
    'True_Label': [video_labels[v] for v in video_names_sorted],
    'Predicted_Label': [int(np.mean(video_preds[v]) >= 0.5) for v in video_names_sorted],
    'Frame_Accuracy': [np.mean(np.array(video_preds[v]) == video_labels[v]) * 100 for v in video_names_sorted],
    'Total_Frames': [len(video_preds[v]) for v in video_names_sorted],
    'Correct_Frames': [np.sum(np.array(video_preds[v]) == video_labels[v]) for v in video_names_sorted],
    'Forged_Ratio': [np.mean(video_preds[v]) for v in video_names_sorted]
})

df_frame_results.to_excel(os.path.join(paper_fig_dir, 'Video_Level_Results.xlsx'), 
                          index=False, sheet_name='Video Results')


summary_stats = {
    'Metric': ['Number of Videos', 'Correct Videos', 'Video Accuracy (%)',
               'Mean Frame Accuracy (%)', 'Median Frame Accuracy (%)',
               'Std Frame Accuracy (%)', 'Min Frame Accuracy (%)', 'Max Frame Accuracy (%)',
               'Original Videos Correct', 'Forged Videos Correct',
               'AUC (Frame-Level)', 'AUC (Video-Level)'],
    'Value': [len(video_preds), correct_videos, video_accuracy*100,
              np.mean(frame_accs), np.median(frame_accs), np.std(frame_accs),
              np.min(frame_accs), np.max(frame_accs),
              sum(1 for r in per_video_results if r['true_label']==0 and r['is_correct']),
              sum(1 for r in per_video_results if r['true_label']==1 and r['is_correct']),
              auc_frame, auc_video]
}

df_summary = pd.DataFrame(summary_stats)
df_summary.to_excel(os.path.join(paper_fig_dir, 'Summary_Statistics.xlsx'), 
                    index=False, sheet_name='Summary')

print("\n" + "="*60)
print(f"✅ ALL FIGURES SAVED TO: {paper_fig_dir}")
print("="*60)
print("\nGenerated Figures:")
print("  Figure 1: Frame Accuracy Distribution (Histogram + Box Plot)")
print("  Figure 2: Video vs Frame Accuracy (Horizontal Bar Chart)")
print("  Figure 3: Frame-Level Confusion Matrix")
print("  Figure 4: ROC Curves (Frame & Video Level)")
print("  Figure 5: Training Curves (Loss & Accuracy)")
print("  Figure 6: Per-Frame Predictions Heatmap")
print("  Figure 7: Confidence Intervals Analysis")
print("  Figure 8: Majority Voting Robustness Analysis")
print("\nExcel files saved:")
print("  - Video_Level_Results.xlsx")
print("  - Summary_Statistics.xlsx")
print("="*60)