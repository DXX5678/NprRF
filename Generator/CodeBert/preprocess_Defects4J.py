import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--buggy_file', help='buggy file path')
    parser.add_argument('--buggy_line', help='buggy code line')
    parser.add_argument('--temp_file', help='temp file path')
    args = parser.parse_args()
    buggy_file_lines = open(args.buggy_file, "r").readlines()
    if "," in args.buggy_line:
        ab_buggy_lines = list(map(int, args.buggy_line.split(",")))
    else:
        ab_buggy_lines = [int(args.buggy_line)]
    output_file = open(args.temp_file, "w")
    input = []
    for line in ab_buggy_lines:
        input.append(buggy_file_lines[line-1])
    output_file.write("".join(input).strip().replace("\n", " "))
    output_file.close()
    
    
if __name__ == "__main__":
    main()