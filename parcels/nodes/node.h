#ifndef _NODE_H
#define _NODE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

typedef struct _NodeJIT {
    void* _c_prev_p;
    //void* _c_self_p;
    void* _c_next_p;
    void* _c_data_p;
    int _c_pu_affinity;
} NodeJIT;

void init_node(NodeJIT* self_node);
void set_prev_ptr(NodeJIT* self_node, NodeJIT* prev_node);
void set_next_ptr(NodeJIT* self_node, NodeJIT* next_node);
void set_data_ptr(NodeJIT* self_node, void* data_ptr);
void set_pu_affinity(NodeJIT* self_node, int pu_number);
int get_pu_affinity(NodeJIT* self_node);
void set_pu_num(NodeJIT* self_node, int pu_number);
int get_pu_num(NodeJIT* self_node);
void reset_prev_ptr(NodeJIT* self_node);
void reset_next_ptr(NodeJIT* self_node);
void reset_data_ptr(NodeJIT* self_node);
void reset_pu_affinity(NodeJIT* self_node);
void reset_pu_number(NodeJIT* self_node);




#ifdef __cplusplus
}
#endif
#endif