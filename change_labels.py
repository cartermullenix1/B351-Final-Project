import os

# Directory containing the label files
labels_dir = 'Helmet_Only_dataset/train/labels'

# Iterate over each file in the directory
for filename in os.listdir(labels_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(labels_dir, filename)
        
        # Read the current contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Process each line and change the label
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            label = parts[0]
            
            # Change the label from 1 to 0 and 0 to 1
            if label == '1':
                parts[0] = '0'
            elif label == '0':
                parts[0] = '1'
            
            new_lines.append(' '.join(parts))
        
        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.write('\n'.join(new_lines))
