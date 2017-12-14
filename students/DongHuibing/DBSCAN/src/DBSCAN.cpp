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
double dis[MAXN][MAXN];
double eps;
int minPts;

void readData()
{
    string s = "";
    cout << "Please enter the file name , eps and minPts:" << endl;
    cin >> s >> eps >> minPts;
    freopen(("../../data/synthetic_data/" + s).c_str(), "r", stdin);
    double x, y;
    int t;
    while(~scanf("%lf,%lf,%d", &x, &y, &t))
    {
        nodes.push_back((Node){x, y, t, -1});
    }
}

double squDistance(Node &u, Node &v)
{
    double dx = u.x - v.x;
    double dy = u.y - v.y;
    return dx * dx + dy * dy;
}

double mydistance(Node &u, Node &v)
{
    return sqrt(squDistance(u, v));
}

void calcDis()
{
    for(int i = 0; i < nodes.size(); ++i)
    {
        for(int j = i + 1; j < nodes.size(); ++j)
        {
            dis[i][j] = dis[j][i] = squDistance(nodes[i], nodes[j]);
        }
    }
}

void dbscan()
{
    int clusterId = 0;
    for(int i = 0; i < nodes.size(); ++i)
    {
        Node &cur = nodes[i];
        if(cur.cluster >= 0) continue;
        queue<int> neighbors;
        set<int> se;
        se.insert(i);
        for(int j = 0; j < nodes.size(); ++j)
        {
            if(i == j || nodes[j].cluster > 0) continue;
            if(mydistance(cur, nodes[j]) <= eps)
            {
                neighbors.push(j);
                se.insert(j);
            }
        }
        if(neighbors.size() < minPts)
        {
            nodes[i].cluster = 0;
            continue;
        }
        nodes[i].cluster = ++clusterId;
        while(!neighbors.empty())
        {
            int tp = neighbors.front();
            neighbors.pop();
            nodes[tp].cluster = clusterId;
            for(int j = 0; j < nodes.size(); ++j)
            {
                if(j == i || j == tp || nodes[j].cluster > 0 || se.find(j) != se.end()) continue;
                if(mydistance(nodes[tp], nodes[j]) <= eps)
                {
                    se.insert(j);
                    neighbors.push(j);
                }
            }
        }
    }
}

void out(vector<Node> &nodes)
{
    for(int i = 0; i < nodes.size(); ++i)
        cout << nodes[i].x << ' ' << nodes[i].y << ' ' << nodes[i].truth << ' ' << nodes[i].cluster << endl;
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
    calcDis();
    dbscan();
    outputResult();
    return 0;
}
