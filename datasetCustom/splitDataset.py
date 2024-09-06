import os
import shutil
import random

def split_dataset(train_val_ratio=0.9):
    current_dir = os.getcwd()  # Ottiene la cartella corrente
    
    # Creiamo le cartelle di output se non esistono
    train_val_dir = os.path.join(current_dir, 'train_val')
    test_dir = os.path.join(current_dir, 'test')
    os.makedirs(train_val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iteriamo su ogni cartella di classe nella cartella corrente
    for class_name in os.listdir(current_dir):
        class_dir = os.path.join(current_dir, class_name)
        if os.path.isdir(class_dir) and class_name not in ['train_val', 'test']:
            # Creiamo le cartelle di classe nelle cartelle di destinazione
            train_val_class_dir = os.path.join(train_val_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_val_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Lista di tutte le immagini .jpg in tutte le sottocartelle della classe
            images = []
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith('.jpg'):  # Controlliamo che il file sia .jpg
                        images.append((os.path.join(root, file), os.path.basename(root)))
            
            random.shuffle(images)  # Mischiamo le immagini

            # Dividiamo le immagini tra train_val e test
            train_val_split = int(len(images) * train_val_ratio)
            train_val_images = images[:train_val_split]
            test_images = images[train_val_split:]

            # Copiamo le immagini nella rispettiva cartella, rinominandole se necessario
            for img_path, subfolder_name in train_val_images:
                img_name = subfolder_name + '_' + os.path.basename(img_path)  # Prefisso basato sul nome della sottocartella
                destination_path = os.path.join(train_val_class_dir, img_name)
                shutil.copy(img_path, destination_path)

            for img_path, subfolder_name in test_images:
                img_name = subfolder_name + '_' + os.path.basename(img_path)  # Prefisso basato sul nome della sottocartella
                destination_path = os.path.join(test_class_dir, img_name)
                shutil.copy(img_path, destination_path)

if __name__ == "__main__":
    split_dataset()
