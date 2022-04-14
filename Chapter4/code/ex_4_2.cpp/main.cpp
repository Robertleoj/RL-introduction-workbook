#include <bits/stdc++.h>

using namespace std;

enum Dir{
    dup = 0,
    dright = 1,
    ddown = 2,
    dleft = 3
};

auto scores = map<int, double>();

void move(int x, Dir d, int *nxt, double *reward){
    *reward = -1;
    if(d == ddown){
        switch(x){
            case 11:
                *nxt = 0;
                return;
            case 14: case 12: case 15:
                *nxt = x;
                return;
            case 13:
                *nxt = 15;
                // *nxt = 13;
                return;
            default:
                *nxt = x + 4;
                return;
        }
    }

    if(d == dright){
        switch(x){
            case 14:
                *nxt = 0;
                return;
            case 11: case 7: case 3: 
                *nxt = x;
                return;
            case 15:
                *nxt = 14;
                return;
            default:
                *nxt = x + 1;
                return;
        }
    }

    if(d == dup){
        switch(x){
            case 1: case 2: case 3:
                *nxt = x;
                return;
            case 4:
                *nxt = 0;
                return;

            case 15:
                *nxt = 13;
                return;
            default:
                *nxt = x - 4;
                return;
        }
    }

    if(d == dleft){
        switch(x){
            case 1: 
                *nxt = 0;
                return;
            case 4: case 8: case 12:
                *nxt = x;
                return;
            case 15:
                *nxt = 12;
                return;
            default:
                *nxt = x - 1;
                return;
        }
    }
}

void print_board(){
    for(int i = 0; i <= 16; i++){

        switch(i){
            case 0:
                cout << scores[0] <<  " ";
                break;
            case 15:
                cout << scores[0] << " " << endl;
                break;
            case 16:
                cout << "  " << scores[15] << endl;
                break;
            default:
                cout << scores[i] << " ";
                break;
        }
        switch(i){
            case 3: case 7: case 11:
                cout << endl;
        }
    }
}


int main(){

    for (int i = 0; i <= 15; i++){
        scores[i] = 0;
    }

    double newscore;
    double rew;
    int x;
    for(int i = 0; i < 1000000; i++){
        for(int j = 1; j <= 15;  j ++){
            newscore = 0;
            for(int dir = 0; dir < 4; dir ++){
                move(j, (Dir)dir,&x, &rew);
                newscore += 0.25 * (rew + scores[x]);
                // cout << scores[x];
                // cout << rew << endl;
            }
            scores[j] = newscore;
        }
    }

    print_board();

}