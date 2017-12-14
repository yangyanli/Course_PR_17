#include<bits/stdc++.h>
#include<unistd.h>
using namespace std;
struct Node
{
    double x, y;
    int truth;
    int cluster;//-1:undefined;0:noise;

    bool operator< (const Node &rhs) const
    {
        if (x != rhs.x) return x < rhs.x;
        return y < rhs.y;
    }
};
vector<Node> nodes;
const int MAXN = 1e4 + 5;
double sim[MAXN][MAXN], deg[MAXN][MAXN], L[MAXN][MAXN];
double sigma;
int k;

void readData()
{
    string s = "";
    cout << "Please enter the file name , sigma and k:" << endl;
    cin >> s >> sigma >> k;
    freopen(("../data/synthetic_data/" + s).c_str(), "r", stdin);
    double x, y;
    int t;
    while(~scanf("%lf,%lf,%d", &x, &y, &t))
        nodes.push_back((Node){x, y, t, -1});
}

double squDistance(Node &u, Node &v)
{
    double dx = u.x - v.x;
    double dy = u.y - v.y;
    return dx * dx + dy * dy;
}


void calcSim()
{
    for(int i = 0; i < nodes.size(); ++i)
        for(int j = i; j < nodes.size(); ++j)
            sim[i][j] = sim[j][i] = exp(-squDistance(nodes[i], nodes[j])*0.5/sigma/sigma);
}

void calcDeg()
{
    for(int i = 0; i < nodes.size(); ++i)
        for(int j = i; j < nodes.size(); ++j) deg[i][j] = deg[j][i] = 0;
    for(int i = 0; i < nodes.size(); ++i)
        for(int j = i + 1; j < nodes.size(); ++j)
        {
            deg[i][i] += sim[i][j];
            deg[j][j] += sim[i][j];
        }
}

void calcLap()
{
    for(int i = 0; i < nodes.size(); ++i)
        for(int j = 0; j < nodes.size(); ++j) L[i][j] = sim[i][j] - deg[i][j];
}

void outputResult()
{
    freopen("../data/data.tsv", "w", stdout);
    puts("x\ty\tclusters");
    for(int i = 0; i < nodes.size(); ++i)
    {
        printf("%.2f\t%.2f\tCluster%d\n", nodes[i].x, nodes[i].y, nodes[i].cluster);
    }
}


int main()
{
    readData();
    calcSim();
    calcDeg();
    calcLap();
    outputResult();
    return 0;
}
