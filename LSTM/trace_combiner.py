import os
import lzma

def process_xz_files(folder_path, output_file_path, start_line=4_000_000, lines_to_collect=2_000_000):
    total_processed_lines = 0
    total_files_processed = 0

    # start enumerating
    with open(output_file_path, 'w') as output_file:

        # Iterate through each file 
        for file_index, file_name in enumerate(os.listdir(folder_path)):
            if file_name.endswith('.xz'):
                file_path = os.path.join(folder_path, file_name)
                print(f"File name: {file_name}")
                try:

                    # Open and process the .xz file
                    with lzma.open(file_path, 'rt') as input_file:  # rt for reading 

                        current_line = 0
                        file_processed_lines = 0

                        for line in input_file:
                            current_line += 1   # iteration number

                            # Skip lines to get a good starting point in file
                            # This number was discussed with the sponsor
                            if current_line < start_line:
                                continue

                            # Stop processing when correct number of lines have been reached
                            if file_processed_lines >= lines_to_collect:
                                print(f"Reached {lines_to_collect} lines for {file_name}. Moving to next file.")
                                break

                            # Process the line
                            parts = line.strip().split()
                            if len(parts) < 2:
                                print(f"Bad line: {line}")
                                continue  # Skip this bad line; easiest solution but could be improved

                            address = int(parts[1])     # second part is the complete address
                            page_offset = address % 4096    
                            page_number = address >> 12

                            # Format as binary and zfill
                            page_number_binary = bin(page_number)[2:].zfill(14)
                            page_offset_binary = bin(page_offset)[2:].zfill(12)

                            # Combine the fields into one long string without spaces
                            combined_output = f"{page_number_binary}{page_offset_binary}\n"

                            # Write to the output file
                            output_file.write(combined_output)
                            file_processed_lines += 1

                            # Track progress every 100,000 lines for each file
                            if file_processed_lines % 1_000_000 == 0:
                                print(f"Processed {file_processed_lines} lines in {file_name}.")

                    total_processed_lines += file_processed_lines
                    total_files_processed += 1
                    print(f"Finished processing {file_name}. Total files processed: {total_files_processed}")

                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")

                # Print the number of files processed so far
                print(f"Files processed so far: {total_files_processed}/{len(os.listdir(folder_path))}")

    print(f"Total processed lines across all files: {total_processed_lines}")
    print(f"Total files processed: {total_files_processed}")


# Example usage
zip_folder_path = '/Users/nathanielbush/Desktop/ECEN403/NeuralNetwork/4TimeStep/Traces/l2_traces'
output_file_path = 'master_dataset.txt'

# list_files_in_folder(zip_folder_path)

process_xz_files(zip_folder_path, output_file_path)
