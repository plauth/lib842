#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: cl2h input.cl output.h variable_name\n";
        exit(EXIT_FAILURE);
    }

    std::ifstream inputFile(argv[1], std::ifstream::in);
    inputFile.exceptions(std::ifstream::badbit);
    std::ofstream outputFile(argv[2], std::ifstream::out);
    outputFile.exceptions(std::ofstream::badbit);

    outputFile << "static const char *" << argv[3] << " = \n\"";

    int c;
    while ((c = inputFile.get()) != EOF) {
        switch (c) {
            case '\n': outputFile << "\\n\"\n\""; break;
            case '\\': outputFile << "\\\\"; break;
            case '\"': outputFile << "\\\""; break;
            default: outputFile << (char)c; break;
        }
    }

    outputFile << "\"\n;\n";

    return EXIT_SUCCESS;
}
