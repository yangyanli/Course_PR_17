#include<bits/stdc++.h>
#include<unistd.h>
using namespace std;
struct Node
{
    double x, y;
    int truth;
    int cluster;

    bool operator< (const Node &rhs) const
    {
        if (x != rhs.x) return x < rhs.x;
        return y < rhs.y;
    }
};
vector<Node> nodes;
vector<Node> means;
double *centerx, *centery;
int k;
void determineK();

void readData()
{
    string s = "";
    cout << "Please enter the file name and the number of clusters(0 for auto selection):" << endl;
    cin >> s >> k;
    freopen(("../../data/synthetic_data/" + s).c_str(), "r", stdin);
    double x, y;
    int t;
    while(~scanf("%lf,%lf,%d", &x, &y, &t))
    {
        nodes.push_back((Node){x, y, t, -1});
    }
    if(k == 0) determineK();
}

void out(vector<Node> &nodes)
{
    for(int i = 0; i < nodes.size(); ++i)
        cout << nodes[i].x << ' ' << nodes[i].y << ' ' << nodes[i].truth << ' ' << nodes[i].cluster << endl;
}


double squDistance(Node &u, Node &v)
{
    double dx = u.x - v.x;
    double dy = u.y - v.y;
    return dx * dx + dy * dy;
}

void chooseNextMean(set<int> &visIds)
{
    double mx = -1;
    int pos = -1;
    for(int i = 0; i < nodes.size(); ++i)
    {
        if(visIds.find(i) != visIds.end()) continue;
        double curSquDis = 0;
        for(vector<Node>::iterator it = means.begin(); it != means.end(); ++it)
            curSquDis += squDistance(nodes[i], *it);
        if(mx < curSquDis)
        {
            mx = curSquDis;
            pos = i;
        }
    }
    if(pos == -1) pos = rand() % nodes.size() - 1;
    visIds.insert(pos);
    means.push_back(nodes[pos]);
}

void initMeans()
{
    set<int> visIds;
    means.clear();
    srand(time(0));
    assert(nodes.size() > 0);
    int t = rand() % nodes.size();
    visIds.insert(t);
    means.push_back(nodes[t]);

    for(int i = 0; i < k - 1; ++i) chooseNextMean(visIds);

    assert(k == means.size());

//    out(means);
}

bool shouldEnd(vector<Node> &nodes1, vector<Node> &nodes2)
{
    assert(nodes1.size() == nodes2.size());

    int diff = 0, tot = nodes1.size();
    for(int i = 0; i < tot; ++i)
    {
        assert(nodes1[i].x == nodes2[i].x);
        assert(nodes1[i].y == nodes2[i].y);
        assert(nodes1[i].x == nodes2[i].x);
        if(nodes1[i].cluster != nodes2[i].cluster) ++diff;
    }
    if((double)diff / tot > 0.01 || diff > 1000) return false;
    return true;
}

void iterate()
{
    int times = 0;
    vector<Node> preNodes;
    centerx = new double[k];
    centery = new double[k];
    while(true)
    {
        ++times;
//        cout << "times = " << times << endl;
        preNodes = nodes;
        for(int i = 0; i < k; ++i)
        {
            centerx[i] = 0;
            centery[i] = 0;
        }
        sort(means.begin(), means.end());
        for(int i = 0; i < nodes.size(); ++i)
        {
            double mn = squDistance(nodes[i], means[0]);
            int cluster = 0;
            for(int j = 1; j < k; ++j)
            {
                double dis = squDistance(nodes[i], means[j]);
                if(mn > dis)
                {
                    mn = dis;
                    cluster = j;
                }
            }
            nodes[i].cluster = cluster;
            centerx[cluster] += nodes[i].x;
            centery[cluster] += nodes[i].y;
        }
        if(shouldEnd(nodes, preNodes)) break;
    }
    delete[] centerx;
    delete[] centery;
//    out(nodes);
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

double getSse()
{
    double res = 0;
    for(int i = 0; i < nodes.size(); ++i)
    {
        res += squDistance(nodes[i], means[nodes[i].cluster]);
    }
    return res;
}

double getScale(double mn, double mx, double cur)
{
    return (cur - mn) / (mx - mn);
}

void determineK()
{
    if(sqrt(nodes.size()) <= 1)
    {
        k = 1;
        return;
    }

    vector<double> sses;
    for(k = 1; k <= sqrt(nodes.size()) + 1; ++k)
    {
        initMeans();
        iterate();
        sses.push_back(getSse());
    }
    for(int i = 0; i < sses.size(); ++i)
//        cout << "sses[" << i << "] = " << sses[i] << endl;
        cout <<  sses[i] << endl;

    vector<double> diffs;
    for(int i = 1; i < sses.size(); ++i) diffs.push_back(sses[i - 1] - sses[i]);
//    for(int i = 0; i < sses.size(); ++i)
//    {
//        cout << i << " to " << i + 1 << ", difference = " << diffs[i] << endl;
//    }

    double suml = 0, sumr = 0;
    for(int i = 0; i < diffs.size(); ++i) sumr += diffs[i];
    double avgl, avgr;
    double mn = 1e30;
    k = 1;
    for(int i = 0; i < diffs.size() - 2; ++i)
    {
        suml += diffs[i];
        sumr -= diffs[i];
        avgl = suml / (i + 1);
        avgr = sumr / (diffs.size() - i - 1);
        double errl = 0, errr = 0;
        for(int j = 0; j <=i; ++j) errl += fabs(diffs[j] - avgl);
        errl /= suml;
        for(int j = i + 1; j < diffs.size(); ++j) errr += fabs(diffs[j] - avgr);
        errr /= sumr;
//        cout << "k = " << (i + 2) << "; sum of err = " << (errl + errr) << endl;
        if(mn > (errl + errr))
        {
            mn = errl + errr;
            k = i + 2;
        }
    }

    cout << "k = " << k << endl;
}

int main()
{
    readData();
    initMeans();
    iterate();
    outputResult();
    return 0;
}
