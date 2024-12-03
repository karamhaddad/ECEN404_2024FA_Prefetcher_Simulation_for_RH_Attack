def process_file(input_file_path, output_file_path):
    previous_block_number = None  # stores previous block number 
    
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:      # open a read and write file 
        for line in input_file:
            parts = line.strip().split()

            # address is the second part of the line
            address = int(parts[1])  

            # Calculate block number (address % 4096, then right shift by 6)
            block_number = (address % 4096) >> 6

            # If there's a previous block number, calculate the block delta
            if previous_block_number is not None:
                block_delta = block_number - previous_block_number      # this value is what will be fed into the network

                # make sure delta is within range of values allowed in future parts
                if block_delta < -63 or block_delta > 63:
                    raise ValueError(f"Block delta {block_delta} is not within bounds.")

                # Decided to use 2's compliment form to feed in machine
                # Research showd it was the best option when using binary for NN's and 
                # it was the easiest to implement
                if block_delta >= 0:
                    delta_binary = bin(block_delta)[2:].zfill(7)  # Positive delta, remove '0b' prefix
                else:
                    delta_binary = bin(2**7 + block_delta)[2:].zfill(7)  # Convert negative delta to 2's complement

                # Write the binary block delta to the output file
                output_file.write(f"{delta_binary}\n")

            # Set the prvious block number
            previous_block_number = block_number

    print(f"Preprocessing done. Block deltas saved to {output_file_path}.")

# Example usage
input_file_path = '/Users/nathanielbush/Desktop/ECEN403/NeuralNetwork/4TimeStep/654.roms_s-293B.champsimtrace.xz_none_cli_0_l2clog.txt'  # Path to input file
output_file_path = '654_block_deltas_output.txt'  # Output file for block deltas
process_file(input_file_path, output_file_path)