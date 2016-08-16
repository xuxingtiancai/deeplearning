#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ext/hash_map>
#include <pthread.h>
using namespace std;
using namespace __gnu_cxx;

static int  max_line_len=1024;
static char *line  = (char *) realloc(line,max_line_len);
char sku_vec_file[256];
int thread_num=0;
double *item_vec=0;
long int sku_size=0;
int vec_size=0;
int top_num=0;
double time_cal_vec=0;
double time_sort=0;

//xuxing-全局锁
pthread_mutex_t the_mutex;

struct SkuDis{
long  int sku_id;
double dis;
};

bool operator>( SkuDis a, SkuDis b ){
	if(a.dis > b.dis){
		return true;
	}
	else if(a.dis < b.dis){
		return false;
	}
	else{
		return a.sku_id > b.sku_id;
	}
}


hash_map<long int,double*> sku_vec_dict(1000000);
static char* readline(FILE *input)
{
    int len; 

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {    
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

int CalUserSKUSimilarty(double *user_vec,size_t vec_size,SkuDis *skudis){
	hash_map<long int ,double*>::iterator iter=sku_vec_dict.begin();
	double res=0;
	int j=0;
    clock_t t1,t2,t3;
    t1=clock();
	for(hash_map<long int ,double*>::iterator iter=sku_vec_dict.begin();
			iter!=sku_vec_dict.end();++iter){
		res=0;
		double *vec2=iter->second;
		double *vec1=user_vec;
		for(size_t i=0;i<vec_size;i=i+1){
			res+=vec1[i]*vec2[i];
		}
		skudis[j].dis=res;
		skudis[j++].sku_id=iter->first;
	}
    t2=clock();
	sort(skudis,skudis+j-1,greater<SkuDis>());
    t3=clock();
    time_cal_vec+=double((t2-t1));
    time_sort+=double((t3-t2));

	return j;
}

int print_top_sku(SkuDis *sku_dis,int len,const char* user_name){
    char out_line[102400];
    int num=snprintf(out_line,256,"%s\t",user_name);
    size_t minlen=top_num < len?top_num:len;
    for(size_t i=0;i<minlen;++i){
        num+=snprintf(out_line+num,256,"%ld:%0.5f\t",sku_dis[i].sku_id,sku_dis[i].dis);
    }
    out_line[num-1]=0;
    //xuxing-打印标准输出
    pthread_mutex_lock(&the_mutex);
    printf("%s\n",out_line);
    pthread_mutex_unlock(&the_mutex);

    return 0;
}

void *CalUserBestSkuThread(void *id) {
    char *file_line=NULL;
    char *next_str=NULL;
    bool flag=false;	
    double *user_vec=new double[vec_size];
    SkuDis skudis[3000000];
    int line_num =0;
    while(1) {
        memset(skudis,0,3000000*sizeof(SkuDis));
        size_t len=0;
        //xuxing-读入全sku列表
        pthread_mutex_lock(&the_mutex);
        int ret = getline(&file_line,&len,stdin);
        pthread_mutex_unlock(&the_mutex);
        
        if(ret == -1) break;
        char *p=strtok_r(file_line," \t",&next_str);
        char *user_name = p;
        memset(user_vec,0,vec_size*sizeof(double));
        flag=false;
        double sum_square=0;
        int error_tag = 0;
        for(int i = 0; i < vec_size; ++i){
            p=strtok_r(NULL," \t",&next_str);
            if(p == NULL) {
                error_tag = 1;
                break;
            }
            double value=atof(p);
            user_vec[i]=value;
            sum_square += value*value;
        }
        //xuxing-成功读入
        if(error_tag) continue;
        line_num++;
        if(line_num % 100 == 0) fprintf(stderr,"DEBUG: thread=%d, line_num=%d\n", id, line_num);
        sum_square=sqrt(sum_square);
        for(int i=0;i<vec_size;++i){
            user_vec[i]/=sum_square;
        }
	    CalUserSKUSimilarty(user_vec,vec_size,skudis);
        print_top_sku(skudis,50000,user_name);	
    }
    fprintf(stderr,"INFO: user_sku  :%ld lines\n", line_num);
    return 0;
}

//读入热卖词集合
int read_sku_vector(double *&item_vec,hash_map<long int,double*> &item_dict,char *item_vec_file,
                    long int &item_size,int & vec_size)
{
    FILE *fin;
    item_size=0;
    vec_size=0;
    fin = fopen(item_vec_file, "r");
    if (fin == NULL) {
        fprintf(stderr, "ERROR: training data file not found!\n");
        exit(1);
    }
    readline(fin);
    char *p=strtok(line," \t");
    char *endptr=NULL;
    item_size=strtod(p,&endptr);
    p=strtok(NULL," \t");
    vec_size=strtod(p,&endptr);

    item_vec = new double[item_size*vec_size];
    memset(item_vec,0,sizeof(double)*item_size*(vec_size));
    long int line_num=0;
    fprintf(stderr, "INFO: item_size:%ld   vec_size:%d\n",item_size,vec_size);
    while(1) {
        if (!readline(fin)){
            break;
        }
        p=strtok(line," \t");

        long int item_id = strtol(p,NULL,10);
        long int begin=line_num*(vec_size);
        double squart_sum=0;
        for(int i=0;i<vec_size;++i){
            p=strtok(NULL," \t");
            double value=atof(p);
            squart_sum+=value*value;
            item_vec[begin+i]=value;
        }
        for(int i=0;i<vec_size;++i){
            item_vec[begin+i]/=sqrt(squart_sum);
        }
        item_dict[item_id]=&item_vec[begin];
        line_num++;
    }
    fprintf(stderr,"INFO: load sku_vec  sku:%ld lines\n", line_num);
    return 0;
}
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      fprintf(stderr, "ERROR: Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv)
{

  if (argc == 1) {
    fprintf(stderr, "gen_user_vec\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "Parameters for training:\n");
    fprintf(stderr, "\t-sku_vec_file <file>\n");
    fprintf(stderr, "\t-thread n\n");

    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "./gen_user_vec -sku_vec_file -thread \n\n");
    return 0;
  }
  // input:ad_sku,sku_vec

  sku_vec_file[0]=0;

    int i=0;
  if ((i = ArgPos((char *)"-ad_sku_vec_file", argc, argv)) > 0) strcpy(sku_vec_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) thread_num = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-top_num", argc, argv)) > 0) top_num = atoi(argv[i + 1]);
  //if ((i = ArgPos((char *)"-vec_size", argc, argv)) > 0) vec_size= atoi(argv[i + 1]);
  fprintf(stderr, "INFO: ad_sku_vec_file:%s\n",sku_vec_file);
  fprintf(stderr, "INFO: thread:%d\n",thread_num);
  //fprintf(stderr, "INFO: vec_size:%d\n",vec_size);
  read_sku_vector(item_vec,sku_vec_dict,sku_vec_file,sku_size,vec_size);
  fprintf(stderr, "INFO: vec_size:%d\n",vec_size);

  pthread_t *pt = (pthread_t *)malloc(thread_num* sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "ERROR: cannot allocate memory for threads\n");
    exit(1);
  }
    
  for (int a = 0; a < thread_num; a++) {
      pthread_create(&pt[a], NULL,CalUserBestSkuThread, (void *)a);
  }
  for (int a = 0; a < thread_num; a++) {
      pthread_join(pt[a], NULL);
  }
  //fprintf(stderr, "INFO: cal vec mutiply time:%f %f\n",time_cal_vec,time_cal_vec/CLOCKS_PER_SEC);
  //fprintf(stderr, "INFO: sort time:%f %f\n",time_sort,time_sort/CLOCKS_PER_SEC);



  return 0;
}
