#include <bits/stdc++.h>
using namespace std;

typedef long double ld;
typedef long long ll;
typedef pair<long double, long double> pld;

#define pb push_back
#define mp make_pair
#define f first
#define s second

ld **transpose(ld **mat, ll n, ll m){
    ld **output = new ld *[m];
    for(ll i = 0; i < m; i++){
        output[i] = new ld[n];
        for(ll j = 0; j < n; j++){
            output[i][j] = mat[j][i];
        }
    }
    return output;
}

ld *dotProduct(ld **mat, ld *v, ll n, ll m){
    ld *output = new ld[n];
    for(ll i = 0; i < n; i++){
        output[i] = 0;
        for(ll j = 0; j < m; j++){
            output[i] += mat[i][j]*v[j];
        }
    }
    return output;
}

ld *softmax(ld *v, ll n){
    ld *output = new ld[n];
    ld total = 0;
    for(ll i = 0; i < n; i++){
        output[i] = exp(v[i]);
        total += output[i];
    }
    for(ll i = 0; i < n; i++){
        output[i] /= total;
    }
    return output;
}

int main(){
    ll vocab_size, num_dim, num_iter, num_pairs;
    ld learning_rate;
    cin >> vocab_size >> num_dim >> learning_rate >> num_iter >> num_pairs;

    vector<pld> word_pairs;
    ll iter_id, input_word, output_word;
    for(ll i = 0; i < num_pairs; i++){
        cin >> iter_id >> input_word >> output_word;
        word_pairs.pb(mp(input_word-1,output_word-1));
    }

    ld **input_weight = new ld *[vocab_size];
    for(ll i = 0; i < vocab_size; i++){
        input_weight[i] = new ld[num_dim];
        for(ll j = 0; j < num_dim; j++){
            input_weight[i][j] = 0.5;
        }
    }

    ld **hidden_weight = new ld *[num_dim];
    for(ll i = 0; i < num_dim; i++){
        hidden_weight[i] = new ld[vocab_size];
        for(ll j = 0; j < vocab_size; j++){
            hidden_weight[i][j] = 0.5;
        }
    }

    for(ll i = 0; i < num_iter; i++){
        for(ll j = 0; j < num_pairs; j++){
            input_word = word_pairs[j].f;
            output_word = word_pairs[j].s;

            ld pos_change = 0, neg_change = 0;

            ld *input_vec = new ld[vocab_size];
            for(ll k = 0; k < vocab_size; k++){
                input_vec[k] = 0;
            }
            input_vec[input_word] = 1;

            ld *h = dotProduct(transpose(input_weight, vocab_size, num_dim),
                input_vec, num_dim, vocab_size);

            ld *output_vec = dotProduct(transpose(hidden_weight, num_dim, vocab_size),
                h, vocab_size, num_dim);

            ld *softmax_vec = softmax(output_vec, vocab_size);

            ld *e = new ld[vocab_size];
            for(ll k = 0; k < vocab_size; k++){
                if(k == output_word){
                    e[k] = softmax_vec[k]-1;
                }
                else{
                    e[k] = softmax_vec[k];
                }
            }

            ld *del_E_w = new ld[num_dim];
            for(ll k = 0; k < num_dim; k++){
                del_E_w[k] = e[output_word]*h[k];
            }

            for(ll k = 0; k < num_dim; k++){
                hidden_weight[k][output_word] -= learning_rate*del_E_w[k];
                if(learning_rate*del_E_w[k] > 0){
                    neg_change++;
                }
                else{
                    pos_change++;
                }
            }

            ld *del_E_h = new ld[num_dim];
            for(ll k = 0; k < num_dim; k++){
                del_E_h[k] = 0;
                for(ll l = 0; l < vocab_size; l++){
                    del_E_h[k] += e[l]*hidden_weight[k][l];
                }
            }

            for(ll l = 0; l < num_dim; l++){
                input_weight[input_word][l] -= learning_rate*del_E_h[l];
                if(learning_rate*del_E_h[l] > 0){
                    neg_change++;
                }
                else{
                    pos_change++;
                }
            }

            cout << (i+1) << " " << (j+1) << " " << neg_change << " " << pos_change << endl;
        }
    }
}
