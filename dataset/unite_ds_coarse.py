import os
import shutil
import uuid  # Importa il modulo uuid per generare nomi univoci

# Definizione dei percorsi delle cartelle
train_folder = './train/'
test_folder = './test/'
val_folder = './val/'
dataset_folder = './train_val/'

# Funzione per copiare i file mantenendo la struttura delle cartelle CoarseGrainedClass/FineGrainedClass
def copy_files(source_folder, dest_folder):
    for coarse_grained_class in os.listdir(source_folder):
        coarse_grained_class_path = os.path.join(source_folder, coarse_grained_class)
        dest_coarse_grained_class_path = os.path.join(dest_folder, coarse_grained_class)
        
        if not os.path.exists(dest_coarse_grained_class_path):
            os.makedirs(dest_coarse_grained_class_path)
        
        for fine_grained_class in os.listdir(coarse_grained_class_path):
            fine_grained_class_path = os.path.join(coarse_grained_class_path, fine_grained_class)
            dest_fine_grained_class_path = os.path.join(dest_coarse_grained_class_path, fine_grained_class)
            
            if os.path.isdir(fine_grained_class_path):
                if not os.path.exists(dest_fine_grained_class_path):
                    os.makedirs(dest_fine_grained_class_path)
                
                for filename in os.listdir(fine_grained_class_path):
                    source_file = os.path.join(fine_grained_class_path, filename)
                    if os.path.isfile(source_file):
                        dest_filename = f"{fine_grained_class}_{str(uuid.uuid4())}.jpg"
                        dest_file = os.path.join(dest_fine_grained_class_path, dest_filename)
                        shutil.copy(source_file, dest_file)
            
            else:  # Se non è una cartella, è un file all'interno di coarse_grained_class_path
                dest_filename = f"{coarse_grained_class}_{str(uuid.uuid4())}.jpg"
                dest_file = os.path.join(dest_coarse_grained_class_path, dest_filename)
                shutil.copy(fine_grained_class_path, dest_file)

# Copia dei file dai dataset train, test e val nella cartella dataset mantenendo la struttura
copy_files(train_folder, dataset_folder)
copy_files(val_folder, dataset_folder)
#copy_files(test_folder, dataset_folder)

print("Unione dei dataset completata!")
