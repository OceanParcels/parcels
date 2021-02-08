/**
 *
 **/
#include "node.h"



void init_node(NodeJIT* self_node) {
    (*self_node)._c_prev_p = NULL;
    (*self_node)._c_next_p = NULL;
    (*self_node)._c_data_p = NULL;
    (*self_node)._c_pu_affinity = -1;
}

void set_prev_ptr(NodeJIT* self_node, NodeJIT* prev_node) {
    (*self_node)._c_prev_p = (void*)prev_node;
}

void set_next_ptr(NodeJIT* self_node, NodeJIT* next_node) {
    (*self_node)._c_next_p = (void*)next_node;
}

void set_data_ptr(NodeJIT* self_node, void* data_ptr) {
    (*self_node)._c_data_p = (void*)data_ptr;
}

void set_pu_affinity(NodeJIT* self_node, int pu_number) {
    (*self_node)._c_pu_affinity = pu_number;
}

int get_pu_affinity(NodeJIT* self_node) {
    return (*self_node)._c_pu_affinity;
}

void set_pu_num(NodeJIT* self_node, int pu_number) {
    set_pu_affinity(self_node, pu_number);
}

int get_pu_num(NodeJIT* self_node) {
    return get_pu_affinity(self_node);
}

void reset_prev_ptr(NodeJIT* self_node) {
    (*self_node)._c_prev_p = NULL;
}

void reset_next_ptr(NodeJIT* self_node) {
    (*self_node)._c_next_p = NULL;
}

void reset_data_ptr(NodeJIT* self_node) {
    (*self_node)._c_data_p = NULL;
}

void reset_pu_affinity(NodeJIT* self_node) {
    (*self_node)._c_pu_affinity = -1;
}

void reset_pu_number(NodeJIT* self_node) {
    reset_pu_affinity(self_node);
}