import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--buggy_file', help='buggy file path')
    parser.add_argument('--temp_file', help='temp file path')
    parser.add_argument('--start', help='method start line number')
    parser.add_argument('--end', help='method end line number')
    args = parser.parse_args()
    buggy_file_lines = open(args.buggy_file, "r").readlines()
    start_line = int(args.start)
    end_line = int(args.end)
    output_file = open(args.temp_file, "w")
    for i in range(start_line - 1, end_line):
        buggy_line = buggy_file_lines[i]
        output_file.write(buggy_line)
    output_file.close()


if __name__ == "__main__":
    main()