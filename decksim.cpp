#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

using namespace std;

int main() {
    const int N = 52;
    const int TRIALS = 10;

    int trickWorks = 0;
    int noMatch = 0;

    random_device rd;
    mt19937 gen(rd());  // seed once

    for (int t = 0; t < TRIALS; t++) {
        vector<int> d1(N);
        iota(d1.begin(), d1.end(), 1);

        vector<int> d2 = d1;

        shuffle(d1.begin(), d1.end(), gen);
        // shuffle(d2.begin(), d2.end(), gen);

        bool matched = false;

        for (int i = 0; i < N; i++) {
            if (d1[i] == d2[i]) {
                matched = true;
                break;
            }
        }

        if (matched)
            trickWorks++;
        else
            noMatch++;
    }

    cout << "Trick works: " << trickWorks << endl;
    cout << "No match: " << noMatch << endl;
}
