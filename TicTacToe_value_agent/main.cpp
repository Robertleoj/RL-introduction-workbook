

#include <stdio.h>
#include <vector>
#include <map>
#include <memory>

#define CIRCLE 0
#define CROSS 1

#define rep(i,b, e) for(int i = b; i < e; i++)

typedef std::pair<int, int> Move;

enum Players{
    Empty,
    Cross, 
    Circle
};

class State{
public:
    Players board[3][3];
    bool turn;
    int moves_made;

    State(){
        moves_made = 0;
        rep(i,0, 3){
            rep(j,0, 3){
                board[i][j] = Empty;
            }
        }
        turn = CROSS;
    }

    void switch_turns(){
        this->turn = !this->turn;
    }

    void is_terminal(bool *is_term, bool *who_won){
        *is_term = false;
        *who_won = Empty;
        if(moves_made == 9){
            *is_term = true;
            return;
        }
        rep(i,0, 3){
            if(board[i][0] != Empty && board[i][0] == board[i][1] && board[i][1] == board[i][2]){
                *is_term = true;
                *who_won = board[i][0];
                return;
            }
            if(board[0][i] != Empty && board[0][i] == board[1][i] && board[1][i] == board[2][i]){
                *is_term = true;
                *who_won = board[0][i];
                return;
            }
        }
        if(board[0][0] != Empty && board[0][0] == board[1][1] && board[1][1] == board[2][2]){
            *is_term = true;
            *who_won = board[0][0];
            return;
        }
        if(board[0][2] != Empty && board[0][2] == board[1][1] && board[1][1] == board[2][0]){
            *is_term = true;
            *who_won = board[0][2];
            return;
        }
    }

    std::vector<Move> available_moves(){
        auto ret = std::vector<Move>();
        rep(i, 0, 3){
            rep(j, 0, 3){
                if(board[i][j] == Empty){
                    ret.push_back(std::make_pair(i, j));
                }
            }
        }
        return ret;
    }

};



class ValueNode{
public:
    std::map<Move, std::unique_ptr<ValueNode>> child_map;
    double value;

    ValueNode(double value){}


};

class ValueTree{
    std::unique_ptr<ValueNode> root;
    ValueNode * curr;

    ValueTree(State * state, double init_value=0.5){
        root = std::unique_ptr<ValueNode>(new ValueNode(0.5));
        curr = root.get();
    }

    void move_and_update(Move our_move, Move opponent_move, double alpha){

    }
};

class Agent{
public:
    Agent(){}
};

