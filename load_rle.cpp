#include "load_rle.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

using namespace std;

static inline size_t idx(int X, int Y)
{
    if (Y >= 0 && X >= 0 && ((Y * CANVAS_SIZE_X + X) < NUM_CELLS - 1)) {
        return (Y * CANVAS_SIZE_X + X);
    } else {
        return NUM_CELLS - 1;
    }
}

vector<string> split(string phrase, string delimiter)
{
    vector<string> list;
    string s = string(phrase);
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    list.push_back(s);
    return list;
}

/** Read file into string. */
inline string slurp(const string& path)
{
    ostringstream buf;
    ifstream input(path.c_str());
    buf << input.rdbuf();
    return buf.str();
}

/** starting from x, set n cells to val**/
void setN(cell* dst, int x, int y, int n, int val)
{
    if (n == 0)
        n = 1;
    debug_print("Setting [%d,%d]->+%d to %d\n", x, y, n, val);
    while (n-- > 0) {
        dst[idx(x + n, y)].x = val;
    }
}
/**
 * Load a standard run-length file, and position it relatively at the given location(center-aligned)
 **/
void load_rle_file(cell* dst, string path, float locx = 0.5, float locy = 0.5)
{
    std::ifstream input(path);
    int width = -1, height = -1;
    int upperx = -1;
    int uppery = -1;
    int x = -1, y = -1;
    string counter = "0";
    for (std::string eachLine; getline(input, eachLine);) {
        //cout << "Parse line: [" << eachLine << "]" << endl;
        if (eachLine[0] == '#')
            cout << eachLine << endl;
        else if (eachLine[0] == 'x') {
            width = stoi(split(split(eachLine, ",")[0], "= ")[1]);
            height = stoi(split(split(eachLine, ",")[1], "= ")[1]);
            cout << "Loading file of width: [" << width << "], height: [" << height << "]" << endl;
        } else {
            if (width <= 0 && height <= 0) {
                debug_print("Invalid file?\n");
                abort();
            }
            if (upperx == -1) { // not the first content line
                upperx = CANVAS_SIZE_X * locx - (float)width * 0.5;
                uppery = CANVAS_SIZE_Y * locy - (float)height * 0.5;
                x = upperx, y = uppery;
                counter = "0";
            }
            for (char c : eachLine) {
                debug_print("%c,", c);
                if (c == 'b') { //dead
                    //setN(dst, x, y, stoi(counter), 0); // not need to set dead cell if zero init
                    x += stoi(counter) == 0 ? 1 : stoi(counter);
                    counter = "0";
                } else if (c == 'o') { // alive
                    setN(dst, x, y, stoi(counter), 1);
                    x += stoi(counter) == 0 ? 1 : stoi(counter);
                    counter = "0";
                } else if (c == '$') { // EOL
                    if (stoi(counter)) {
                        //debug_print("File is bad: counter [%s], line %d: [%s]\n", counter.c_str(), eachLine.c_str());
                        cout << "Bad file, counter: " << counter << ", Line: " << eachLine << endl;
                        //abort();
                    }
                    //y += 1;
                    y += stoi(counter) == 0 ? 1 : stoi(counter);
                    x = upperx;
                    counter = "0";
                    debug_print("---\n");
                } else if (isdigit(c)) { // digits
                    counter += c;
                }
            }
        }
    }
}
