#include "load_rle.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

using namespace std;

static inline ssize_t idx(ssize_t X, ssize_t Y)
{
    if (Y >= 0 && X >= 0 && ((Y * CANVAS_SIZE_X + X) < NUM_CELLS - 1)) {
        //debug_print("idx: (%zd,%zd) -> %zu\n", X, Y, Y * CANVAS_SIZE_X + X);
        return (ssize_t)(Y * CANVAS_SIZE_X + X);
    } else {
        return (ssize_t)NUM_CELLS - 1;
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
void setN(cell* dst, ssize_t x, ssize_t y, ssize_t n, ssize_t val)
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
    ssize_t width = -1, height = -1;
    ssize_t upperx = -1;
    ssize_t uppery = -1;
    ssize_t x = -1, y = -1;
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
                // Place by center
                upperx = CANVAS_SIZE_X * locx - (float)width * 0.5;
                uppery = CANVAS_SIZE_Y * locy - (float)height * 0.5;
                // Place by upper left corner
                //upperx = CANVAS_SIZE_X * locx;
                //uppery = CANVAS_SIZE_Y * locy;
                debug_print("Upper X: %zd, Upper Y: %zd\n", upperx, uppery);
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
