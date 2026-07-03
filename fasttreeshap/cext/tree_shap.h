/**
 * Fast recursive computation of SHAP values in trees.
 * See https://arxiv.org/abs/1802.03888 for details.
 * Scott Lundberg, 2018 (independent algorithm courtesy of Hugh Chen 2018)
 *
 * Fast TreeSHAP algorithm v1 and Fast TreeSHAP algorithm v2.
 * See https://arxiv.org/abs/2109.09847 for details.
 * Jilei Yang, 2021
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h> 
#include <cmath>
#include <ctime>
#if defined(_WIN32) || defined(WIN32)
    #include <malloc.h>
#elif defined(__MVS__)
    #include <stdlib.h>
#else
    #include <alloca.h>
#endif
using namespace std;

typedef double tfloat;
typedef tfloat (* transform_f)(const tfloat margin, const tfloat y);

namespace FEATURE_DEPENDENCE {
    const unsigned independent = 0;
    const unsigned tree_path_dependent = 1;
    const unsigned global_path_dependent = 2;
}

namespace ALGORITHM {
    const unsigned v0 = 0;
    const unsigned v1 = 1;
    const unsigned v2 = 2;
    const unsigned v2_1 = 3;
    const unsigned v2_2 = 4;
    const unsigned v3 = 6;   // batched descent: one traversal per tree, samples as vectors
}

struct TreeEnsemble {
    int *children_left;
    int *children_right;
    int *children_default;
    int *features;
    tfloat *thresholds;
    tfloat *values;
    tfloat *node_sample_weights;
    unsigned max_depth;
    unsigned tree_limit;
    tfloat *base_offset;
    unsigned max_nodes;
    unsigned num_outputs;

    TreeEnsemble() {}
    TreeEnsemble(int *children_left, int *children_right, int *children_default, int *features,
                 tfloat *thresholds, tfloat *values, tfloat *node_sample_weights,
                 unsigned max_depth, unsigned tree_limit, tfloat *base_offset,
                 unsigned max_nodes, unsigned num_outputs) :
        children_left(children_left), children_right(children_right),
        children_default(children_default), features(features), thresholds(thresholds),
        values(values), node_sample_weights(node_sample_weights),
        max_depth(max_depth), tree_limit(tree_limit),
        base_offset(base_offset), max_nodes(max_nodes), num_outputs(num_outputs) {}

    void get_tree(TreeEnsemble &tree, const unsigned i) const {
        const unsigned d = i * max_nodes;

        tree.children_left = children_left + d;
        tree.children_right = children_right + d;
        tree.children_default = children_default + d;
        tree.features = features + d;
        tree.thresholds = thresholds + d;
        tree.values = values + d * num_outputs;
        tree.node_sample_weights = node_sample_weights + d;
        tree.max_depth = max_depth;
        tree.tree_limit = 1;
        tree.base_offset = base_offset;
        tree.max_nodes = max_nodes;
        tree.num_outputs = num_outputs;
    }

    bool is_leaf(unsigned pos)const {
        return children_left[pos] < 0;
    }

    void allocate(unsigned tree_limit_in, unsigned max_nodes_in, unsigned num_outputs_in) {
        tree_limit = tree_limit_in;
        max_nodes = max_nodes_in;
        num_outputs = num_outputs_in;
        children_left = new int[tree_limit * max_nodes];
        children_right = new int[tree_limit * max_nodes];
        children_default = new int[tree_limit * max_nodes];
        features = new int[tree_limit * max_nodes];
        thresholds = new tfloat[tree_limit * max_nodes];
        values = new tfloat[tree_limit * max_nodes * num_outputs];
        node_sample_weights = new tfloat[tree_limit * max_nodes];
    }

    void free() {
        delete[] children_left;
        delete[] children_right;
        delete[] children_default;
        delete[] features;
        delete[] thresholds;
        delete[] values;
        delete[] node_sample_weights;
    }
};

struct ExplanationDataset {
    tfloat *X;
    bool *X_missing;
    tfloat *y;
    tfloat *R;
    bool *R_missing;
    unsigned num_X;
    unsigned M;
    unsigned num_R;

    ExplanationDataset() {}
    ExplanationDataset(tfloat *X, bool *X_missing, tfloat *y, tfloat *R, bool *R_missing, unsigned num_X,
                       unsigned M, unsigned num_R) : 
        X(X), X_missing(X_missing), y(y), R(R), R_missing(R_missing), num_X(num_X), M(M), num_R(num_R) {}

    void get_x_instance(ExplanationDataset &instance, const unsigned i) const {
        instance.M = M;
        instance.X = X + i * M;
        instance.X_missing = X_missing + i * M;
        instance.num_X = 1;
    }
};


// data we keep about our decision path
// note that pweight is included for convenience and is not tied with the other attributes
// the pweight of the i'th path element is the permuation weight of paths with i-1 ones in them
struct PathElement {
    int feature_index;
    tfloat zero_fraction;
    tfloat one_fraction;
    tfloat pweight;
    PathElement() {}
    PathElement(int i, tfloat z, tfloat o, tfloat w) :
        feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
};

inline tfloat logistic_transform(const tfloat margin, const tfloat y) {
    return 1 / (1 + exp(-margin));
}

inline tfloat logistic_nlogloss_transform(const tfloat margin, const tfloat y) {
    return log(1 + exp(margin)) - y * margin; // y is in {0, 1}
}

inline tfloat squared_loss_transform(const tfloat margin, const tfloat y) {
    return (margin - y) * (margin - y);
}

namespace MODEL_TRANSFORM {
    const unsigned identity = 0;
    const unsigned logistic = 1;
    const unsigned logistic_nlogloss = 2;
    const unsigned squared_loss = 3;
}

inline transform_f get_transform(unsigned model_transform) {
    transform_f transform = NULL;
    switch (model_transform) {
        case MODEL_TRANSFORM::logistic:
            transform = logistic_transform;
            break;

        case MODEL_TRANSFORM::logistic_nlogloss:
            transform = logistic_nlogloss_transform;
            break;

        case MODEL_TRANSFORM::squared_loss:
            transform = squared_loss_transform;
            break;
    }

    return transform;
}

inline tfloat *tree_predict(unsigned i, const TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
    const unsigned offset = i * trees.max_nodes;
    unsigned node = 0;
    while (true) {
        const unsigned pos = offset + node;
        const unsigned feature = trees.features[pos];
        
        // we hit a leaf so return a pointer to the values
        if (trees.is_leaf(pos)) {
            return trees.values + pos * trees.num_outputs;
        }
        
        // otherwise we are at an internal node and need to recurse
        if (x_missing[feature]) {
            node = trees.children_default[pos];
        } else if (x[feature] <= trees.thresholds[pos]) {
            node = trees.children_left[pos];
        } else {
            node = trees.children_right[pos];
        }
    }
}

inline void dense_tree_predict(tfloat *out, const TreeEnsemble &trees, const ExplanationDataset &data, unsigned model_transform) {
    tfloat *row_out = out;
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    // see what transform (if any) we have
    transform_f transform = get_transform(model_transform);

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the base offset
        for (unsigned k = 0; k < trees.num_outputs; ++k) {
            row_out[k] += trees.base_offset[k];
        }

        // add the leaf values from each tree
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            const tfloat *leaf_value = tree_predict(j, trees, x, x_missing);

            for (unsigned k = 0; k < trees.num_outputs; ++k) {
                row_out[k] += leaf_value[k];
            }
        }

        // apply any needed transform
        if (transform != NULL) {
            const tfloat y_i = data.y == NULL ? 0 : data.y[i];
            for (unsigned k = 0; k < trees.num_outputs; ++k) {
                row_out[k] = transform(row_out[k], y_i);
            }
        }

        x += data.M;
        x_missing += data.M;
        row_out += trees.num_outputs;
    }
}

inline void tree_update_weights(unsigned i, TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
    const unsigned offset = i * trees.max_nodes;
    unsigned node = 0;
    while (true) {
        const unsigned pos = offset + node;
        const unsigned feature = trees.features[pos];

        // Record that a sample passed through this node
        trees.node_sample_weights[pos] += 1.0;
        
        // we hit a leaf so return a pointer to the values
        if (trees.children_left[pos] < 0) break;
        
        // otherwise we are at an internal node and need to recurse
        if (x_missing[feature]) {
            node = trees.children_default[pos];
        } else if (x[feature] <= trees.thresholds[pos]) {
            node = trees.children_left[pos];
        } else {
            node = trees.children_right[pos];
        }
    }
}

inline void dense_tree_update_weights(TreeEnsemble &trees, const ExplanationDataset &data) {
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the leaf values from each tree
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            tree_update_weights(j, trees, x, x_missing);
        }

        x += data.M;
        x_missing += data.M;
    }
}

inline void tree_saabas(tfloat *out, const TreeEnsemble &tree, const ExplanationDataset &data) {
    unsigned curr_node = 0;
    unsigned next_node = 0;
    while (true) {
        
        // we hit a leaf and are done
        if (tree.children_left[curr_node] < 0) return;
        
        // otherwise we are at an internal node and need to recurse
        const unsigned feature = tree.features[curr_node];
        if (data.X_missing[feature]) {
            next_node = tree.children_default[curr_node];
        } else if (data.X[feature] <= tree.thresholds[curr_node]) {
            next_node = tree.children_left[curr_node];
        } else {
            next_node = tree.children_right[curr_node];
        }

        // assign credit to this feature as the difference in values at the current node vs. the next node
        for (unsigned i = 0; i < tree.num_outputs; ++i) {
            out[feature * tree.num_outputs + i] += tree.values[next_node * tree.num_outputs + i] - tree.values[curr_node * tree.num_outputs + i];
        }

        curr_node = next_node;
    }
}

/**
 * This runs Tree SHAP with a per tree path conditional dependence assumption.
 */
inline void dense_tree_saabas(tfloat *out_contribs, const TreeEnsemble& trees, const ExplanationDataset &data) {
    tfloat *instance_out_contribs;
    TreeEnsemble tree;
    ExplanationDataset instance;

    // build explanation for each sample
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
        data.get_x_instance(instance, i);

        // aggregate the effect of explaining each tree
        // (this works because of the linearity property of Shapley values)
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            trees.get_tree(tree, j);
            tree_saabas(instance_out_contribs, tree, instance);
        }

        // apply the base offset to the bias term
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset[j];
        }
    }
}


// extend our decision path with a fraction of one and zero extensions
inline void extend_path(PathElement *unique_path, unsigned unique_depth,
                        tfloat zero_fraction, tfloat one_fraction, int feature_index) {
    unique_path[unique_depth].feature_index = feature_index;
    unique_path[unique_depth].zero_fraction = zero_fraction;
    unique_path[unique_depth].one_fraction = one_fraction;
    unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
    for (int i = unique_depth - 1; i >= 0; i--) {
        unique_path[i + 1].pweight += one_fraction * unique_path[i].pweight * (i + 1)
                                      / static_cast<tfloat>(unique_depth + 1);
        unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i)
                                 / static_cast<tfloat>(unique_depth + 1);
    }
}

// undo a previous extension of the decision path
inline void unwind_path(PathElement *unique_path, unsigned unique_depth, unsigned path_index) {
    const tfloat one_fraction = unique_path[path_index].one_fraction;
    const tfloat zero_fraction = unique_path[path_index].zero_fraction;
    tfloat next_one_portion = unique_path[unique_depth].pweight;

    for (int i = unique_depth - 1; i >= 0; --i) {
        if (one_fraction != 0) {
            const tfloat tmp = unique_path[i].pweight;
            unique_path[i].pweight = next_one_portion * (unique_depth + 1)
                                     / static_cast<tfloat>((i + 1) * one_fraction);
            next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i)
                               / static_cast<tfloat>(unique_depth + 1);
        } else {
            unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1))
                                     / static_cast<tfloat>(zero_fraction * (unique_depth - i));
        }
    }

    for (unsigned i = path_index; i < unique_depth; ++i) {
        unique_path[i].feature_index = unique_path[i+1].feature_index;
        unique_path[i].zero_fraction = unique_path[i+1].zero_fraction;
        unique_path[i].one_fraction = unique_path[i+1].one_fraction;
    }
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path
inline tfloat unwound_path_sum(const PathElement *unique_path, unsigned unique_depth,
                               unsigned path_index) {
    const tfloat one_fraction = unique_path[path_index].one_fraction;
    const tfloat zero_fraction = unique_path[path_index].zero_fraction;
    tfloat next_one_portion = unique_path[unique_depth].pweight;
    tfloat total = 0;

    if (one_fraction != 0) {
        for (int i = unique_depth - 1; i >= 0; --i) {
            const tfloat tmp = next_one_portion / static_cast<tfloat>((i + 1) * one_fraction);
            total += tmp;
            next_one_portion = unique_path[i].pweight - tmp * zero_fraction * (unique_depth - i);
        }
    } else {
        for (int i = unique_depth - 1; i >= 0; --i) {
            total += unique_path[i].pweight / (zero_fraction * (unique_depth - i));
        }
    }
    return total * (unique_depth + 1);
}

// recursive computation of SHAP values for a decision tree
inline void tree_shap_recursive(const unsigned num_outputs, const int *children_left,
                                const int *children_right,
                                const int *children_default, const int *features,
                                const tfloat *thresholds, const tfloat *values,
                                const tfloat *node_sample_weight,
                                const tfloat *x, const bool *x_missing, tfloat *phi,
                                unsigned node_index, unsigned unique_depth,
                                PathElement *parent_unique_path, tfloat parent_zero_fraction,
                                tfloat parent_one_fraction, int parent_feature_index,
                                int condition, unsigned condition_feature,
                                tfloat condition_fraction) {

    // stop if we have no weight coming down to us
    if (condition_fraction == 0) return;

    // extend the unique path
    PathElement *unique_path = parent_unique_path + unique_depth + 1;
    std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);

    if (condition == 0 || condition_feature != static_cast<unsigned>(parent_feature_index)) {
        extend_path(unique_path, unique_depth, parent_zero_fraction,
                    parent_one_fraction, parent_feature_index);
    }
    const unsigned split_index = features[node_index];

    // leaf node
    if (children_right[node_index] < 0) {
        for (unsigned i = 1; i <= unique_depth; ++i) {
            const tfloat w = unwound_path_sum(unique_path, unique_depth, i);
            const PathElement &el = unique_path[i];
            const unsigned phi_offset = el.feature_index * num_outputs;
            const unsigned values_offset = node_index * num_outputs;
            const tfloat scale = w * (el.one_fraction - el.zero_fraction) * condition_fraction;
            for (unsigned j = 0; j < num_outputs; ++j) {
                phi[phi_offset + j] += scale * values[values_offset + j];
            }
        }

    // internal node
    } else {
        // find which branch is "hot" (meaning x would follow it)
        unsigned hot_index = 0;
        if (x_missing[split_index]) {
            hot_index = children_default[node_index];
        } else if (x[split_index] <= thresholds[node_index]) {
            hot_index = children_left[node_index];
        } else {
            hot_index = children_right[node_index];
        }
        const unsigned cold_index = (static_cast<int>(hot_index) == children_left[node_index] ?
                                        children_right[node_index] : children_left[node_index]);
        const tfloat w = node_sample_weight[node_index];
        const tfloat hot_zero_fraction = node_sample_weight[hot_index] / w;
        const tfloat cold_zero_fraction = node_sample_weight[cold_index] / w;
        tfloat incoming_zero_fraction = 1;
        tfloat incoming_one_fraction = 1;

        // see if we have already split on this feature,
        // if so we undo that split so we can redo it for this node
        unsigned path_index = 0;
        for (; path_index <= unique_depth; ++path_index) {
            if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_index) break;
        }
        if (path_index != unique_depth + 1) {
            incoming_zero_fraction = unique_path[path_index].zero_fraction;
            incoming_one_fraction = unique_path[path_index].one_fraction;
            unwind_path(unique_path, unique_depth, path_index);
            unique_depth -= 1;
        }

        // divide up the condition_fraction among the recursive calls
        tfloat hot_condition_fraction = condition_fraction;
        tfloat cold_condition_fraction = condition_fraction;
        if (condition > 0 && split_index == condition_feature) {
            cold_condition_fraction = 0;
            unique_depth -= 1;
        } else if (condition < 0 && split_index == condition_feature) {
            hot_condition_fraction *= hot_zero_fraction;
            cold_condition_fraction *= cold_zero_fraction;
            unique_depth -= 1;
        }

        tree_shap_recursive(
            num_outputs, children_left, children_right, children_default, features, thresholds, values,
            node_sample_weight, x, x_missing, phi, hot_index, unique_depth + 1, unique_path,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition, condition_feature, hot_condition_fraction
        );

        tree_shap_recursive(
            num_outputs, children_left, children_right, children_default, features, thresholds, values,
            node_sample_weight, x, x_missing, phi, cold_index, unique_depth + 1, unique_path,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition, condition_feature, cold_condition_fraction
        );
    }
}

inline int compute_expectations(TreeEnsemble &tree, int i = 0, int depth = 0) {
    unsigned max_depth = 0;

    if (tree.children_right[i] >= 0) {
        const unsigned li = tree.children_left[i];
        const unsigned ri = tree.children_right[i];
        const unsigned depth_left = compute_expectations(tree, li, depth + 1);
        const unsigned depth_right = compute_expectations(tree, ri, depth + 1);
        const tfloat left_weight = tree.node_sample_weights[li];
        const tfloat right_weight = tree.node_sample_weights[ri];
        const unsigned li_offset = li * tree.num_outputs;
        const unsigned ri_offset = ri * tree.num_outputs;
        const unsigned i_offset = i * tree.num_outputs;
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            if ((left_weight == 0) && (right_weight == 0)) {
                tree.values[i_offset + j] = 0.0;
            } else {
                const tfloat v = (left_weight * tree.values[li_offset + j] + right_weight * tree.values[ri_offset + j]) / (left_weight + right_weight);
                tree.values[i_offset + j] = v;
            }
        }
        max_depth = std::max(depth_left, depth_right) + 1;
    }
    
    if (depth == 0) tree.max_depth = max_depth;
    
    return max_depth;
}

inline void tree_shap(const TreeEnsemble& tree, const ExplanationDataset &data,
                      tfloat *out_contribs, int condition, unsigned condition_feature) {

    // update the reference value with the expected value of the tree's predictions
    if (condition == 0) {
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            out_contribs[data.M * tree.num_outputs + j] += tree.values[j];
        }
    }

    // Pre-allocate space for the unique path data
    const unsigned maxd = tree.max_depth + 2; // need a bit more space than the max depth
    PathElement *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];

    tree_shap_recursive(
        tree.num_outputs, tree.children_left, tree.children_right, tree.children_default,
        tree.features, tree.thresholds, tree.values, tree.node_sample_weights, data.X,
        data.X_missing, out_contribs, 0, 0, unique_path_data, 1, 1, -1, condition,
        condition_feature, 1
    );

    delete[] unique_path_data;
}


// extend our decision path with a fraction of one and zero extensions
// update unique_path and pweights for the feature of the last split
inline void extend_path_v1(PathElement *unique_path, tfloat *pweights, unsigned unique_depth,
                           unsigned unique_depth_pweights, tfloat zero_fraction, tfloat one_fraction,
                           int feature_index) {
    unique_path[unique_depth].feature_index = feature_index;
    unique_path[unique_depth].zero_fraction = zero_fraction;
    unique_path[unique_depth].one_fraction = one_fraction;
    if (one_fraction != 0) {
        // extend pweights iff the feature of the last split satisfies the threshold
        pweights[unique_depth_pweights] = (unique_depth_pweights == 0 ? 1.0f : 0.0f);
        for (int i = unique_depth_pweights - 1; i >= 0; i--) {
            pweights[i + 1] += pweights[i] * (i + 1) / static_cast<tfloat>(unique_depth + 1);
            pweights[i] *= zero_fraction * (unique_depth - i) / static_cast<tfloat>(unique_depth + 1);
        }
    } else {
        for (int i = unique_depth_pweights - 1; i >= 0; i--) {
            pweights[i] *= (unique_depth - i) / static_cast<tfloat>(unique_depth + 1);
        }
    }
}

// undo a previous extension of the decision path
inline void unwind_path_v1(PathElement *unique_path, tfloat *pweights, unsigned unique_depth,
                           unsigned unique_depth_pweights, unsigned path_index) {
    const tfloat one_fraction = unique_path[path_index].one_fraction;
    const tfloat zero_fraction = unique_path[path_index].zero_fraction;
    tfloat next_one_portion = pweights[unique_depth_pweights];

    if (one_fraction != 0) {
        // shrink pweights iff the feature satisfies the threshold
        for (int i = unique_depth_pweights - 1; i >= 0; --i) {
            const tfloat tmp = pweights[i];
            pweights[i] = next_one_portion * (unique_depth + 1) / static_cast<tfloat>(i + 1);
            next_one_portion = tmp - pweights[i] * zero_fraction * (unique_depth - i)
                               / static_cast<tfloat>(unique_depth + 1);
        }
    } else {
        for (int i = unique_depth_pweights; i >= 0; --i) {
            pweights[i] *= (unique_depth + 1) / static_cast<tfloat>(unique_depth - i);
        }
    }

    for (unsigned i = path_index; i < unique_depth; ++i) {
        unique_path[i].feature_index = unique_path[i+1].feature_index;
        unique_path[i].zero_fraction = unique_path[i+1].zero_fraction;
        unique_path[i].one_fraction = unique_path[i+1].one_fraction;
    }
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path (for feature satisfying the threshold)
inline tfloat unwound_path_sum_v1(const PathElement *unique_path, const tfloat *pweights, unsigned unique_depth,
                                  unsigned unique_depth_pweights, unsigned path_index) {
    tfloat total = 0;
    const tfloat zero_fraction = unique_path[path_index].zero_fraction;
    tfloat next_one_portion = pweights[unique_depth_pweights];
    for (int i = unique_depth_pweights - 1; i >= 0; --i) {
        const tfloat tmp = next_one_portion / static_cast<tfloat>(i + 1);
        total += tmp;
        next_one_portion = pweights[i] - tmp * zero_fraction * (unique_depth - i);
    }
    return total * (unique_depth + 1);
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path (for features not satisfying the thresholds)
inline tfloat unwound_path_sum_zero_v1(const tfloat *pweights, unsigned unique_depth, unsigned unique_depth_pweights) {
    tfloat total = 0;
    if (unique_depth > unique_depth_pweights) {
         for (int i = unique_depth_pweights; i >= 0; --i) {
             total += pweights[i] / static_cast<tfloat>(unique_depth - i);
         }
    }
    return total * (unique_depth + 1);
}


// recursive computation of SHAP values for a decision tree
inline void tree_shap_recursive_v1(const unsigned num_outputs, const int *children_left,
                                   const int *children_right,
                                   const int *children_default, const int *features,
                                   const tfloat *thresholds, const tfloat *values,
                                   const tfloat *node_sample_weight,
                                   const tfloat *x, const bool *x_missing, tfloat *phi,
                                   unsigned node_index, unsigned unique_depth, unsigned unique_depth_pweights,
                                   PathElement *parent_unique_path, tfloat *parent_pweights,
                                   tfloat pweights_residual, tfloat parent_zero_fraction,
                                   tfloat parent_one_fraction, int parent_feature_index,
                                   int condition, unsigned condition_feature,
                                   tfloat condition_fraction) {

    // stop if we have no weight coming down to us
    if (condition_fraction == 0) return;

    // extend the unique path
    PathElement *unique_path = parent_unique_path + unique_depth + 1;
    std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);
    tfloat *pweights = parent_pweights + unique_depth_pweights + 1;
    std::copy(parent_pweights, parent_pweights + unique_depth_pweights + 1, pweights);

    if (condition == 0 || condition_feature != static_cast<unsigned>(parent_feature_index)) {
        extend_path_v1(unique_path, pweights, unique_depth, unique_depth_pweights, parent_zero_fraction,
                       parent_one_fraction, parent_feature_index);
        // update pweights_residual iff the feature of the last split does not satisfy the threshold
        if (parent_one_fraction != 1) {
            pweights_residual *= parent_zero_fraction;
            unique_depth_pweights -= 1;
        }
    }
    const unsigned split_index = features[node_index];

    // leaf node
    if (children_right[node_index] < 0) {
        const unsigned values_offset = node_index * num_outputs;
        unsigned values_nonzero_ind = 0;
        unsigned values_nonzero_count = 0;
        for (unsigned j = 0; j < num_outputs; ++j) {
            if (values[values_offset + j] != 0) {
                values_nonzero_ind = j;
                values_nonzero_count++;
            }
        }
        // pre-calculate w_zero for all features not satisfying the thresholds
        const tfloat w_zero = unwound_path_sum_zero_v1(pweights, unique_depth, unique_depth_pweights);
        const tfloat scale_zero = -w_zero * pweights_residual * condition_fraction;
        tfloat scale;
        for (unsigned i = 1; i <= unique_depth; ++i) {
            const PathElement &el = unique_path[i];
            const unsigned phi_offset = el.feature_index * num_outputs;
            // update contributions to SHAP values for features satisfying the thresholds and not satisfying the thresholds separately
            if (el.one_fraction != 0) {
                const tfloat w = unwound_path_sum_v1(unique_path, pweights, unique_depth, unique_depth_pweights, i);
                scale = w * pweights_residual * (1 - el.zero_fraction) * condition_fraction;
            } else {
                scale = scale_zero;
            }
            if (values_nonzero_count == 1) {
                phi[phi_offset + values_nonzero_ind] += scale * values[values_offset + values_nonzero_ind];
            } else {
                for (unsigned j = 0; j < num_outputs; ++j) {
                    phi[phi_offset + j] += scale * values[values_offset + j];
                }
            }
        }

    // internal node
    } else {
        // find which branch is "hot" (meaning x would follow it)
        unsigned hot_index = 0;
        if (x_missing[split_index]) {
            hot_index = children_default[node_index];
        } else if (x[split_index] <= thresholds[node_index]) {
            hot_index = children_left[node_index];
        } else {
            hot_index = children_right[node_index];
        }
        const unsigned cold_index = (static_cast<int>(hot_index) == children_left[node_index] ?
                                        children_right[node_index] : children_left[node_index]);
        const tfloat w = node_sample_weight[node_index];
        const tfloat hot_zero_fraction = node_sample_weight[hot_index] / w;
        const tfloat cold_zero_fraction = node_sample_weight[cold_index] / w;
        tfloat incoming_zero_fraction = 1;
        tfloat incoming_one_fraction = 1;

        // see if we have already split on this feature,
        // if so we undo that split so we can redo it for this node
        unsigned path_index = 0;
        for (; path_index <= unique_depth; ++path_index) {
            if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_index) break;
        }
        if (path_index != unique_depth + 1) {
            incoming_zero_fraction = unique_path[path_index].zero_fraction;
            incoming_one_fraction = unique_path[path_index].one_fraction;
            unwind_path_v1(unique_path, pweights, unique_depth, unique_depth_pweights, path_index);
            unique_depth -= 1;
            // update pweights_residual iff the duplicated feature does not satisfy the threshold
            if (incoming_one_fraction != 0.) {
                unique_depth_pweights -= 1;
            } else {
                pweights_residual /= incoming_zero_fraction;
            }
        }

        // divide up the condition_fraction among the recursive calls
        tfloat hot_condition_fraction = condition_fraction;
        tfloat cold_condition_fraction = condition_fraction;
        if (condition > 0 && split_index == condition_feature) {
            cold_condition_fraction = 0;
            unique_depth -= 1;
            unique_depth_pweights -= 1;
        } else if (condition < 0 && split_index == condition_feature) {
            hot_condition_fraction *= hot_zero_fraction;
            cold_condition_fraction *= cold_zero_fraction;
            unique_depth -= 1;
            unique_depth_pweights -= 1;
        }

        tree_shap_recursive_v1(
            num_outputs, children_left, children_right, children_default, features, thresholds, values,
            node_sample_weight, x, x_missing, phi, hot_index, unique_depth + 1, unique_depth_pweights + 1,
            unique_path, pweights, pweights_residual,
            hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
            split_index, condition, condition_feature, hot_condition_fraction
        );

        tree_shap_recursive_v1(
            num_outputs, children_left, children_right, children_default, features, thresholds, values,
            node_sample_weight, x, x_missing, phi, cold_index, unique_depth + 1, unique_depth_pweights + 1,
            unique_path, pweights, pweights_residual,
            cold_zero_fraction * incoming_zero_fraction, 0,
            split_index, condition, condition_feature, cold_condition_fraction
        );
    }
}


inline void tree_shap_v1(const TreeEnsemble& tree, const ExplanationDataset &data,
                         tfloat *out_contribs, int condition, unsigned condition_feature) {

    // update the reference value with the expected value of the tree's predictions
    if (condition == 0) {
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            out_contribs[data.M * tree.num_outputs + j] += tree.values[j];
        }
    }

    // pre-allocate space for the unique path data and pweights
    const unsigned maxd = tree.max_depth + 2; // need a bit more space than the max depth
    PathElement *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];
    tfloat *pweights = new tfloat[(maxd * (maxd + 1)) / 2];

    tree_shap_recursive_v1(
        tree.num_outputs, tree.children_left, tree.children_right, tree.children_default,
        tree.features, tree.thresholds, tree.values, tree.node_sample_weights, data.X,
        data.X_missing, out_contribs, 0, 0, 0, unique_path_data, pweights, 1, 1, 1, -1, condition,
        condition_feature, 1
    );

    delete[] unique_path_data;
    delete[] pweights;
}


// recursive computation of combination_sum (matrix S) for a decision tree
inline void compute_combination_sum_recursive_v2(const int *children_left, const int *children_right,
                                                 const int *features, const tfloat *node_sample_weight,
                                                 const int max_depth, tfloat *combination_sum, int *duplicated_node,
                                                 unsigned node_index, unsigned unique_depth,
                                                 int *parent_unique_depth_pweights, PathElement *parent_unique_path,
                                                 tfloat *parent_pweights, tfloat parent_zero_fraction,
                                                 int parent_feature_index, int *leaf_count) {

    // extend the unique path
    PathElement *unique_path = parent_unique_path + unique_depth;
    std::copy(parent_unique_path, parent_unique_path + unique_depth, unique_path);
    unique_path[unique_depth].feature_index = parent_feature_index;
    unique_path[unique_depth].zero_fraction = parent_zero_fraction;

    unsigned l;
    int *unique_depth_pweights;
    tfloat *pweights;
    tfloat *t_pweights;

    // extend pweights and update unique_depth_pweights
    if (unique_depth == 0) {
        l = 1;
        unique_depth_pweights = parent_unique_depth_pweights;
        unique_depth_pweights[0] = 0;
        pweights = parent_pweights;
        pweights[0] = 1.0f;
    }
    else {
        l = static_cast<int>(1 << (unique_depth - 1));
        unique_depth_pweights = parent_unique_depth_pweights + l;
        std::copy(parent_unique_depth_pweights, parent_unique_depth_pweights + l, unique_depth_pweights);
        std::copy(parent_unique_depth_pweights, parent_unique_depth_pweights + l, unique_depth_pweights + l);
        pweights = parent_pweights + l * (max_depth + 1);
        std::copy(parent_pweights, parent_pweights + l * (max_depth + 1), pweights);
        std::copy(parent_pweights, parent_pweights + l * (max_depth + 1), pweights + l * (max_depth + 1));

        for (unsigned t = 0; t < l; t++) {
            t_pweights = pweights + t * (max_depth + 1);
            for (int i = unique_depth_pweights[t] - 1; i >= 0; i--) {
                t_pweights[i] *= (unique_depth - i) / static_cast<tfloat>(unique_depth + 1);
            }
            unique_depth_pweights[t] -= 1;
        }
        for (unsigned t = l; t < 2 * l; t++) {
            t_pweights = pweights + t * (max_depth + 1);
            t_pweights[unique_depth_pweights[t]] = 0.0f;
            for (int i = unique_depth_pweights[t] - 1; i >= 0; i--) {
                t_pweights[i + 1] += t_pweights[i] * (i + 1) / static_cast<tfloat>(unique_depth + 1);
                t_pweights[i] *= parent_zero_fraction * (unique_depth - i) / static_cast<tfloat>(unique_depth + 1);
            }
        }
    }

    const unsigned split_index = features[node_index];

    // leaf node
    if (children_right[node_index] < 0) {
        // calculate one row of combination_sum for the current path
        tfloat *leaf_combination_sum = combination_sum + leaf_count[0] * static_cast<int>(1 << max_depth);
        for (unsigned t = 0; t < 2 * l - 1; t++) {
            leaf_combination_sum[t] = 0;
            t_pweights = pweights + t * (max_depth + 1);
            for (int i = unique_depth_pweights[t]; i >= 0; i--) {
                leaf_combination_sum[t] += t_pweights[i] / static_cast<tfloat>(unique_depth - i);
            }
            leaf_combination_sum[t] *= (unique_depth + 1);
        }
        leaf_count[0] += 1;
    // internal node
    } else {
        const unsigned left_index = children_left[node_index];
        const unsigned right_index = children_right[node_index];
        const tfloat w = node_sample_weight[node_index];
        const tfloat left_zero_fraction = node_sample_weight[left_index] / w;
        const tfloat right_zero_fraction = node_sample_weight[right_index] / w;
        tfloat incoming_zero_fraction = 1;

        // see if we have already split on this feature,
        // if so we undo that split so we can redo it for this node
        unsigned path_index = 0;
        for (; path_index <= unique_depth; ++path_index) {
            if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_index) break;
        }
        if (path_index != unique_depth + 1) {
            duplicated_node[node_index] = path_index;  // record node index of duplicated feature
            incoming_zero_fraction = unique_path[path_index].zero_fraction;

            // shrink pweights and unique_path, and update unique_depth_pweights, given the duplicated feature
            unsigned p = static_cast<int>(1 << (path_index - 1));
            unsigned t = 0;
            tfloat *k_pweights;
            for (unsigned j = 0; j < 2 * l; j += 2 * p) {
                for (unsigned k = j; k < j + p; k++) {
                    t_pweights = pweights + t * (max_depth + 1);
                    k_pweights = pweights + k * (max_depth + 1);
                    for (int i = unique_depth_pweights[k]; i >= 0; i--) {
                        t_pweights[i] = k_pweights[i] * (unique_depth + 1) / static_cast<tfloat>(unique_depth - i);
                    }
                    unique_depth_pweights[t] = unique_depth_pweights[k];
                    t += 1;
                }
            }
            for (unsigned i = path_index; i < unique_depth; ++i) {
                unique_path[i].feature_index = unique_path[i + 1].feature_index;
                unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
            }
            unique_depth -= 1;
        } else {
            duplicated_node[node_index] = -1;
        }

        for (unsigned t = 0; t < 2 * l; t++) {
            unique_depth_pweights[t] += 1;
        }

        compute_combination_sum_recursive_v2(
            children_left, children_right, features, node_sample_weight, max_depth, combination_sum,
            duplicated_node, left_index, unique_depth + 1, unique_depth_pweights, unique_path, pweights,
            incoming_zero_fraction * left_zero_fraction, split_index, leaf_count
        );

        compute_combination_sum_recursive_v2(
            children_left, children_right, features, node_sample_weight, max_depth, combination_sum,
            duplicated_node, right_index, unique_depth + 1, unique_depth_pweights, unique_path, pweights,
            incoming_zero_fraction * right_zero_fraction, split_index, leaf_count
        );
    }
}


// computation of combination_sum (matrix S) for a decision tree
inline void compute_combination_sum_v2(const TreeEnsemble& tree, tfloat *combination_sum, int *duplicated_node) {

    // Pre-allocate space for the unique path data, pweights and unique_depth_pweights
    const unsigned max_combinations = static_cast<int>(1 << tree.max_depth);
    int *unique_depth_pweights = new int[2 * max_combinations];
    tfloat *pweights = new tfloat[2 * max_combinations * (tree.max_depth + 1)];
    PathElement *unique_path_data = new PathElement[(tree.max_depth + 1) * (tree.max_depth + 2) / 2];
    int *leaf_count = new int[1];
    leaf_count[0] = 0;

    compute_combination_sum_recursive_v2(
        tree.children_left, tree.children_right, tree.features, tree.node_sample_weights, tree.max_depth,
        combination_sum, duplicated_node, 0, 0, unique_depth_pweights, unique_path_data, pweights, 1, -1, leaf_count
    );

    delete[] unique_depth_pweights;
    delete[] pweights;
    delete[] unique_path_data;
    delete[] leaf_count;
}


// loop-invariant inputs of the recursive v2 traversal, packed so each recursive
// call passes a single pointer instead of ~14 unchanging arguments
struct TreeShapV2Context {
    unsigned num_outputs;
    const int *children_left;
    const int *children_right;
    const int *features;
    const tfloat *thresholds;
    const tfloat *values;
    const tfloat *node_sample_weight;
    int max_depth;
    const tfloat *combination_sum;
    const int *duplicated_node;
    const tfloat *x;
    tfloat *phi;
    PathElement *unique_path;
    int *leaf_count;
};

// recursive computation of SHAP values for a decision tree
inline void tree_shap_recursive_v2(const TreeShapV2Context &ctx,
                                   unsigned node_index, unsigned unique_depth,
                                   tfloat pweights_residual,
                                   tfloat parent_zero_fraction, tfloat parent_one_fraction,
                                   int parent_feature_index, unsigned combination_sum_ind) {
    const unsigned num_outputs = ctx.num_outputs;
    const tfloat *values = ctx.values;
    tfloat *phi = ctx.phi;
    int *leaf_count = ctx.leaf_count;

    // extend the unique path in place; instead of copying the parent path at every
    // node (O(depth) per node) we mutate the shared array and undo any prefix
    // modification (duplicated-element removal) before returning
    PathElement *unique_path = ctx.unique_path;
    unique_path[unique_depth].feature_index = parent_feature_index;
    unique_path[unique_depth].zero_fraction = parent_zero_fraction;
    unique_path[unique_depth].one_fraction = parent_one_fraction;
    // maintain combination_sum_ind incrementally: the element appended at index
    // unique_depth contributes bit (unique_depth - 1) when its one_fraction is nonzero
    if (unique_depth > 0 && parent_one_fraction != 0) {
        combination_sum_ind += 1u << (unique_depth - 1);
    }
    // update pweights_residual iff the feature of the last split does not satisfy the threshold
    if (parent_one_fraction != 1) {
        pweights_residual *= parent_zero_fraction;
    }

    const unsigned split_index = ctx.features[node_index];

    // leaf node
    if (ctx.children_right[node_index] < 0) {
        const tfloat *leaf_combination_sum = ctx.combination_sum + leaf_count[0] * (1 << ctx.max_depth);
        // update contributions to SHAP values for features satisfying the thresholds and not satisfying the thresholds separately
        const unsigned values_offset = node_index * num_outputs;
        unsigned values_nonzero_ind = 0;
        unsigned values_nonzero_count = 0;
        for (unsigned j = 0; j < num_outputs; ++j) {
            if (values[values_offset + j] != 0) {
                values_nonzero_ind = j;
                values_nonzero_count++;
            }
        }
        const tfloat scale_zero = -leaf_combination_sum[combination_sum_ind] * pweights_residual;
        for (unsigned i = 1; i <= unique_depth; ++i) {
            const PathElement &el = unique_path[i];
            const unsigned phi_offset = el.feature_index * num_outputs;
            const tfloat scale = (el.one_fraction != 0) ? leaf_combination_sum[combination_sum_ind - (1 << (i - 1))] * \
            pweights_residual * (1 - el.zero_fraction) : scale_zero;
            if (values_nonzero_count == 1) {
                phi[phi_offset + values_nonzero_ind] += scale * values[values_offset + values_nonzero_ind];
            } else {
                for (unsigned j = 0; j < num_outputs; ++j) {
                    phi[phi_offset + j] += scale * values[values_offset + j];
                }
            }
        }
        leaf_count[0] += 1;
    // internal node
    } else {
        const unsigned left_index = ctx.children_left[node_index];
        const unsigned right_index = ctx.children_right[node_index];
        const tfloat w = ctx.node_sample_weight[node_index];
        const tfloat left_zero_fraction = ctx.node_sample_weight[left_index] / w;
        const tfloat right_zero_fraction = ctx.node_sample_weight[right_index] / w;
        tfloat incoming_zero_fraction = 1;
        tfloat incoming_one_fraction = 1;

        // see if we have already split on this feature,
        // if so we undo that split so we can redo it for this node
        const int path_index = ctx.duplicated_node[node_index];
        PathElement removed_element;
        if (path_index >= 0) {
            incoming_zero_fraction = unique_path[path_index].zero_fraction;
            incoming_one_fraction = unique_path[path_index].one_fraction;
            removed_element = unique_path[path_index];

            for (unsigned i = path_index; i < unique_depth; ++i) {
                unique_path[i].feature_index = unique_path[i + 1].feature_index;
                unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
                unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
            }
            unique_depth -= 1;
            // remove the duplicated element's bit (path_index - 1) and shift higher bits down
            if (path_index > 0) {
                const unsigned low_mask = (1u << (path_index - 1)) - 1;
                combination_sum_ind = (combination_sum_ind & low_mask) |
                                      ((combination_sum_ind >> path_index) << (path_index - 1));
            } else {
                combination_sum_ind >>= 1;
            }
            // update pweights_residual iff the duplicated feature does not satisfy the threshold
            if (incoming_one_fraction != 1.) {
                pweights_residual /= incoming_zero_fraction;
            }
        }

        tree_shap_recursive_v2(
            ctx, left_index, unique_depth + 1, pweights_residual,
            left_zero_fraction * incoming_zero_fraction,
            incoming_one_fraction * int(ctx.x[split_index] <= ctx.thresholds[node_index]),
            split_index, combination_sum_ind
        );

        tree_shap_recursive_v2(
            ctx, right_index, unique_depth + 1, pweights_residual,
            right_zero_fraction * incoming_zero_fraction,
            incoming_one_fraction * int(ctx.x[split_index] > ctx.thresholds[node_index]),
            split_index, combination_sum_ind
        );

        // undo the duplicated-element removal so the caller's shared path prefix is restored
        if (path_index >= 0) {
            for (int i = (int)unique_depth; i >= path_index; --i) {
                unique_path[i + 1] = unique_path[i];
            }
            unique_path[path_index] = removed_element;
        }
    }
}


inline void tree_shap_v2(const TreeEnsemble& tree, const tfloat *combination_sum, const int *duplicated_node,
                         const ExplanationDataset &data, tfloat *out_contribs) {

    // update the reference value with the expected value of the tree's predictions
    for (unsigned j = 0; j < tree.num_outputs; ++j) {
        out_contribs[data.M * tree.num_outputs + j] += tree.values[j];
    }

    // pre-allocate space for the unique path data
    PathElement *unique_path_data = new PathElement[(tree.max_depth + 1) * (tree.max_depth + 2) / 2];
    int *leaf_count = new int[1];
    leaf_count[0] = 0;

    TreeShapV2Context ctx;
    ctx.num_outputs = tree.num_outputs;
    ctx.children_left = tree.children_left;
    ctx.children_right = tree.children_right;
    ctx.features = tree.features;
    ctx.thresholds = tree.thresholds;
    ctx.values = tree.values;
    ctx.node_sample_weight = tree.node_sample_weights;
    ctx.max_depth = tree.max_depth;
    ctx.combination_sum = combination_sum;
    ctx.duplicated_node = duplicated_node;
    ctx.x = data.X;
    ctx.phi = out_contribs;
    ctx.unique_path = unique_path_data;
    ctx.leaf_count = leaf_count;

    tree_shap_recursive_v2(ctx, 0, 0, 1, 1, 1, -1, 0);

    delete[] unique_path_data;
    delete[] leaf_count;
}


// ---------------------------------------------------------------------------
// "v3" batched descent
//
// v0/v1/v2 all re-walk the tree once per sample. But along any root-to-leaf
// path the path structure (features, zero fractions, duplicate removals) is
// sample-independent — the only per-sample state is one pass/fail bit per
// split and a running residual. So v3 walks each tree ONCE, carrying every
// sample through the traversal as vectors: the subset-index bits (K), the
// pweights residual (R), and the branch bits (B) are arrays over samples,
// updated with dense loops instead of a recursion per sample. It applies the
// identical per-sample arithmetic in the identical DFS order as v2, so the
// output is bit-identical; the win is amortized traversal, branch-free inner
// loops, and cache-friendly access. Complexity matches v2: O(TL2^D + MTLD).
// ---------------------------------------------------------------------------

struct BatchedPathElement {
    int feature_index;
    tfloat zero_fraction;
};

struct TreeShapV3Context {
    unsigned num_X;              // samples in the batch
    unsigned num_outputs;
    const int *children_left;
    const int *children_right;
    const int *features;
    const tfloat *thresholds;
    const tfloat *values;
    const tfloat *node_sample_weight;
    int max_depth;
    const tfloat *combination_sum;
    const int *duplicated_node;
    const tfloat *Xt;            // transposed inputs, feature-major: Xt[f * num_X + m]
    tfloat *phi_t;               // transposed phi: phi_t[(f * num_outputs + j) * num_X + m]
    unsigned *K;                 // per recursion level: subset-index bits per sample
    tfloat *R;                   // per recursion level: pweights residual per sample
    unsigned char *B;            // per recursion level: branch bit per sample
    unsigned char *INC;          // per recursion level: incoming (duplicate) bit per sample
    BatchedPathElement *path;    // triangular scalar path buffer (copied per node; once per tree)
    int *leaf_count;
};

inline void tree_shap_recursive_v3(const TreeShapV3Context &ctx, unsigned node_index,
                                   unsigned row, unsigned unique_depth,
                                   BatchedPathElement *parent_path,
                                   tfloat parent_zero_fraction, int parent_feature_index) {
    const unsigned num_X = ctx.num_X;
    unsigned *K_row = ctx.K + row * num_X;
    tfloat *R_row = ctx.R + row * num_X;

    // extend the scalar path (copy-on-descend: runs once per node per TREE, not per sample)
    BatchedPathElement *path = parent_path + unique_depth;
    for (unsigned i = 0; i < unique_depth; ++i) path[i] = parent_path[i];
    path[unique_depth].feature_index = parent_feature_index;
    path[unique_depth].zero_fraction = parent_zero_fraction;

    // extend the per-sample vectors from the parent level
    if (row == 0) {
        for (unsigned m = 0; m < num_X; ++m) { K_row[m] = 0; R_row[m] = 1; }
    } else {
        const unsigned *K_par = ctx.K + (row - 1) * num_X;
        const tfloat *R_par = ctx.R + (row - 1) * num_X;
        const unsigned char *B_row = ctx.B + row * num_X;
        const unsigned bitpos = unique_depth - 1;
        const tfloat zf = parent_zero_fraction;
        for (unsigned m = 0; m < num_X; ++m) {
            const unsigned b = B_row[m];
            K_row[m] = K_par[m] | (b << bitpos);                 // bit add iff one_fraction != 0
            R_row[m] = b ? R_par[m] : R_par[m] * zf;             // residual mult iff one_fraction != 1
        }
    }

    // leaf node
    if (ctx.children_right[node_index] < 0) {
        const tfloat *S = ctx.combination_sum + ctx.leaf_count[0] * (1 << ctx.max_depth);
        const unsigned values_offset = node_index * ctx.num_outputs;
        for (unsigned i = 1; i <= unique_depth; ++i) {
            const unsigned f = path[i].feature_index;
            const unsigned bm = 1u << (i - 1);
            const tfloat c1 = 1 - path[i].zero_fraction;
            for (unsigned j = 0; j < ctx.num_outputs; ++j) {
                const tfloat v = ctx.values[values_offset + j];
                if (v == 0) continue;                            // adding 0.0 is a no-op; skip the pass
                tfloat *phi_f = ctx.phi_t + (f * ctx.num_outputs + j) * num_X;
                for (unsigned m = 0; m < num_X; ++m) {
                    const unsigned k = K_row[m];
                    const tfloat scale = (k & bm) ? S[k - bm] * R_row[m] * c1
                                                  : -S[k] * R_row[m];
                    phi_f[m] += scale * v;
                }
            }
        }
        ctx.leaf_count[0] += 1;
        return;
    }

    // internal node
    const unsigned split_index = ctx.features[node_index];
    const unsigned left_index = ctx.children_left[node_index];
    const unsigned right_index = ctx.children_right[node_index];
    const tfloat w = ctx.node_sample_weight[node_index];
    const tfloat left_zero_fraction = ctx.node_sample_weight[left_index] / w;
    const tfloat right_zero_fraction = ctx.node_sample_weight[right_index] / w;
    tfloat incoming_zero_fraction = 1;

    unsigned char *INC_row = ctx.INC + row * num_X;
    const int path_index = ctx.duplicated_node[node_index];
    bool has_dup = (path_index >= 0);
    if (has_dup) {
        incoming_zero_fraction = path[path_index].zero_fraction;
        const unsigned inc_bm = 1u << (path_index - 1);

        // capture the per-sample incoming bit BEFORE splicing it out of K
        for (unsigned m = 0; m < num_X; ++m) INC_row[m] = (K_row[m] & inc_bm) ? 1 : 0;

        // remove the duplicated element from the scalar path (local copy — no undo needed)
        for (unsigned i = path_index; i < unique_depth; ++i) path[i] = path[i + 1];
        unique_depth -= 1;

        // splice the duplicated bit out of every sample's subset index
        const unsigned low_mask = inc_bm - 1;
        for (unsigned m = 0; m < num_X; ++m) {
            const unsigned k = K_row[m];
            K_row[m] = (k & low_mask) | ((k >> path_index) << (path_index - 1));
        }
        // residual division where the sample failed the duplicated split
        const tfloat izf = incoming_zero_fraction;
        for (unsigned m = 0; m < num_X; ++m) {
            if (!INC_row[m]) R_row[m] /= izf;
        }
    }

    const tfloat *x_col = ctx.Xt + split_index * num_X;
    const tfloat thr = ctx.thresholds[node_index];
    unsigned char *B_child = ctx.B + (row + 1) * num_X;

    // left child: branch bit = incoming bit AND (x <= threshold)
    if (has_dup) {
        for (unsigned m = 0; m < num_X; ++m) B_child[m] = INC_row[m] & (unsigned char)(x_col[m] <= thr);
    } else {
        for (unsigned m = 0; m < num_X; ++m) B_child[m] = (unsigned char)(x_col[m] <= thr);
    }
    tree_shap_recursive_v3(ctx, left_index, row + 1, unique_depth + 1, path,
                           left_zero_fraction * incoming_zero_fraction, split_index);

    // right child: branch bit = incoming bit AND (x > threshold)
    if (has_dup) {
        for (unsigned m = 0; m < num_X; ++m) B_child[m] = INC_row[m] & (unsigned char)(x_col[m] > thr);
    } else {
        for (unsigned m = 0; m < num_X; ++m) B_child[m] = (unsigned char)(x_col[m] > thr);
    }
    tree_shap_recursive_v3(ctx, right_index, row + 1, unique_depth + 1, path,
                           right_zero_fraction * incoming_zero_fraction, split_index);
}


inline unsigned build_merged_tree_recursive(TreeEnsemble &out_tree, const TreeEnsemble &trees,
                                     const tfloat *data, const bool *data_missing, int *data_inds,
                                     const unsigned num_background_data_inds, unsigned num_data_inds,
                                     unsigned M, unsigned row = 0, unsigned i = 0, unsigned pos = 0,
                                     tfloat *leaf_value = NULL) {
    //tfloat new_leaf_value[trees.num_outputs];
    tfloat *new_leaf_value = (tfloat *) alloca(sizeof(tfloat) * trees.num_outputs); // allocate on the stack
    unsigned row_offset = row * trees.max_nodes;
  
    // we have hit a terminal leaf!!!
    if (trees.children_left[row_offset + i] < 0 && row + 1 == trees.tree_limit) {

        // create the leaf node
        const tfloat *vals = trees.values + (row * trees.max_nodes + i) * trees.num_outputs;
        if (leaf_value == NULL) {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                out_tree.values[pos * trees.num_outputs + j] = vals[j];
            }
        } else {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                out_tree.values[pos * trees.num_outputs + j] = leaf_value[j] + vals[j];
            }
        }
        out_tree.children_left[pos] = -1;
        out_tree.children_right[pos] = -1;
        out_tree.children_default[pos] = -1;
        out_tree.features[pos] = -1;
        out_tree.thresholds[pos] = 0;
        out_tree.node_sample_weights[pos] = num_background_data_inds;

        return pos;
    }
  
    // we hit an intermediate leaf (so just add the value to our accumulator and move to the next tree)
    if (trees.children_left[row_offset + i] < 0) {
        
        // accumulate the value of this original leaf so it will land on all eventual terminal leaves
        const tfloat *vals = trees.values + (row * trees.max_nodes + i) * trees.num_outputs;
        if (leaf_value == NULL) {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                new_leaf_value[j] = vals[j];
            }
        } else {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                new_leaf_value[j] = leaf_value[j] + vals[j];
            }
        }
        leaf_value = new_leaf_value;

        // move forward to the next tree
        row += 1;
        row_offset += trees.max_nodes;
        i = 0;
    }
    
    // split the data inds by this node's threshold
    const tfloat t = trees.thresholds[row_offset + i];
    const int f = trees.features[row_offset + i];
    const bool right_default = trees.children_default[row_offset + i] == trees.children_right[row_offset + i];
    int low_ptr = 0;
    int high_ptr = num_data_inds - 1;
    unsigned num_left_background_data_inds = 0;
    int low_data_ind;
    while (low_ptr <= high_ptr) {
        low_data_ind = data_inds[low_ptr];
        const int data_ind = std::abs(low_data_ind) * M + f;
        const bool is_missing = data_missing[data_ind];
        if ((!is_missing && data[data_ind] > t) || (right_default && is_missing)) {
            data_inds[low_ptr] = data_inds[high_ptr];
            data_inds[high_ptr] = low_data_ind;
            high_ptr -= 1;
        } else {
            if (low_data_ind >= 0) ++num_left_background_data_inds; // negative data_inds are not background samples
            low_ptr += 1;
        }
    }
    int *left_data_inds = data_inds;
    const unsigned num_left_data_inds = low_ptr;
    int *right_data_inds = data_inds + low_ptr;
    const unsigned num_right_data_inds = num_data_inds - num_left_data_inds;
    const unsigned num_right_background_data_inds = num_background_data_inds - num_left_background_data_inds;
  
    // all the data went right, so we skip creating this node and just recurse right
    if (num_left_data_inds == 0) {
        return build_merged_tree_recursive(
            out_tree, trees, data, data_missing, data_inds,
            num_background_data_inds, num_data_inds, M, row,
            trees.children_right[row_offset + i], pos, leaf_value
        );

    // all the data went left, so we skip creating this node and just recurse left
    } else if (num_right_data_inds == 0) {
        return build_merged_tree_recursive(
            out_tree, trees, data, data_missing, data_inds,
            num_background_data_inds, num_data_inds, M, row,
            trees.children_left[row_offset + i], pos, leaf_value
        );

    // data went both ways so we create this node and recurse down both paths
    } else {
        
        // build the left subtree
        const unsigned new_pos = build_merged_tree_recursive(
            out_tree, trees, data, data_missing, left_data_inds,
            num_left_background_data_inds, num_left_data_inds, M, row,
            trees.children_left[row_offset + i], pos + 1, leaf_value
        );

        // fill in the data for this node
        out_tree.children_left[pos] = pos + 1;
        out_tree.children_right[pos] = new_pos + 1;
        if (trees.children_left[row_offset + i] == trees.children_default[row_offset + i]) {
            out_tree.children_default[pos] = pos + 1;
        } else {
            out_tree.children_default[pos] = new_pos + 1;
        }
        
        out_tree.features[pos] = trees.features[row_offset + i];
        out_tree.thresholds[pos] = trees.thresholds[row_offset + i];
        out_tree.node_sample_weights[pos] = num_background_data_inds;

        // build the right subtree
        return build_merged_tree_recursive(
            out_tree, trees, data, data_missing, right_data_inds,
            num_right_background_data_inds, num_right_data_inds, M, row,
            trees.children_right[row_offset + i], new_pos + 1, leaf_value
        );
    }
}


inline void build_merged_tree(TreeEnsemble &out_tree, const ExplanationDataset &data, const TreeEnsemble &trees) {
    
    // create a joint data matrix from both X and R matrices
    tfloat *joined_data = new tfloat[(data.num_X + data.num_R) * data.M];
    std::copy(data.X, data.X + data.num_X * data.M, joined_data);
    std::copy(data.R, data.R + data.num_R * data.M, joined_data + data.num_X * data.M);
    bool *joined_data_missing = new bool[(data.num_X + data.num_R) * data.M];
    std::copy(data.X_missing, data.X_missing + data.num_X * data.M, joined_data_missing);
    std::copy(data.R_missing, data.R_missing + data.num_R * data.M, joined_data_missing + data.num_X * data.M);

    // create an starting array of data indexes we will recursively sort
    int *data_inds = new int[data.num_X + data.num_R];
    for (unsigned i = 0; i < data.num_X; ++i) data_inds[i] = i;
    for (unsigned i = data.num_X; i < data.num_X + data.num_R; ++i) {
        data_inds[i] = -i; // a negative index means it won't be recorded as a background sample
    }

    build_merged_tree_recursive(
        out_tree, trees, joined_data, joined_data_missing, data_inds, data.num_R,
        data.num_X + data.num_R, data.M
    );

    delete[] joined_data;
    delete[] joined_data_missing;
    delete[] data_inds;
}


// Independent Tree SHAP functions below here
// ------------------------------------------
struct Node {
    short cl, cr, cd, pnode, feat, pfeat; // uint_16
    float thres, value;
    char from_flag;
};

#define FROM_NEITHER 0
#define FROM_X_NOT_R 1
#define FROM_R_NOT_X 2

// https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
inline int bin_coeff(int n, int k) { 
    int res = 1; 
    if (k > n - k)
        k = n - k; 
    for (int i = 0; i < k; ++i) { 
        res *= (n - i); 
        res /= (i + 1); 
    } 
    return res; 
} 

// note this only handles single output models, so multi-output models get explained using multiple passes
inline void tree_shap_indep(const unsigned max_depth, const unsigned num_feats,
                            const unsigned num_nodes, const tfloat *x,
                            const bool *x_missing, const tfloat *r,
                            const bool *r_missing, tfloat *out_contribs,
                            float *pos_lst, float *neg_lst, signed short *feat_hist,
                            float *memoized_weights, int *node_stack, Node *mytree) {

//     const bool DEBUG = true;
//     ofstream myfile;
//     if (DEBUG) {
//       myfile.open ("/homes/gws/hughchen/shap/out.txt",fstream::app);
//       myfile << "Entering tree_shap_indep\n";
//     }
    int ns_ctr = 0;
    std::fill_n(feat_hist, num_feats, 0);
    short node = 0, feat, cl, cr, cd, pnode, pfeat = -1;
    short next_xnode = -1, next_rnode = -1;
    short next_node = -1, from_child = -1;
    float thres, pos_x = 0, neg_x = 0, pos_r = 0, neg_r = 0;
    char from_flag;
    unsigned M = 0, N = 0;
    
    Node curr_node = mytree[node];
    feat = curr_node.feat;
    thres = curr_node.thres;
    cl = curr_node.cl;
    cr = curr_node.cr;
    cd = curr_node.cd;

    // short circut when this is a stump tree (with no splits)
    if (cl < 0) {
        out_contribs[num_feats] += curr_node.value;
        return;
    }
    
//     if (DEBUG) {
//       myfile << "\nNode: " << node << "\n";
//       myfile << "x[feat]: " << x[feat] << ", r[feat]: " << r[feat] << "\n";
//       myfile << "thres: " << thres << "\n";
//     }
    
    if (x_missing[feat]) {
        next_xnode = cd;
    } else if (x[feat] > thres) {
        next_xnode = cr;
    } else if (x[feat] <= thres) {
        next_xnode = cl;
    }
    
    if (r_missing[feat]) {
        next_rnode = cd;
    } else if (r[feat] > thres) {
        next_rnode = cr;
    } else if (r[feat] <= thres) {
        next_rnode = cl;
    }
    
    if (next_xnode != next_rnode) {
        mytree[next_xnode].from_flag = FROM_X_NOT_R;
        mytree[next_rnode].from_flag = FROM_R_NOT_X;
    } else {
        mytree[next_xnode].from_flag = FROM_NEITHER;
    }
    
    // Check if x and r go the same way
    if (next_xnode == next_rnode) {
        next_node = next_xnode;
    }
    
    // If not, go left
    if (next_node < 0) {
        next_node = cl;
        if (next_rnode == next_node) { // rpath
            N = N+1;
            feat_hist[feat] -= 1;
        } else if (next_xnode == next_node) { // xpath
            M = M+1;
            N = N+1;
            feat_hist[feat] += 1;
        }
    }
    node_stack[ns_ctr] = node;
    ns_ctr += 1;
    while (true) {
        node = next_node;
        curr_node = mytree[node];
        feat = curr_node.feat;
        thres = curr_node.thres;
        cl = curr_node.cl;
        cr = curr_node.cr;
        cd = curr_node.cd;
        pnode = curr_node.pnode;
        pfeat = curr_node.pfeat;
        from_flag = curr_node.from_flag;

        
        
//         if (DEBUG) {
//           myfile << "\nNode: " << node << "\n";
//           myfile << "N: " << N << ", M: " << M << "\n";
//           myfile << "from_flag==FROM_X_NOT_R: " << (from_flag==FROM_X_NOT_R) << "\n";
//           myfile << "from_flag==FROM_R_NOT_X: " << (from_flag==FROM_R_NOT_X) << "\n";
//           myfile << "from_flag==FROM_NEITHER: " << (from_flag==FROM_NEITHER) << "\n";
//           myfile << "feat_hist[feat]: " << feat_hist[feat] << "\n";
//         }
        
        // At a leaf
        if (cl < 0) {
            //      if (DEBUG) {
            //        myfile << "At a leaf\n";
            //      }

            if (M == 0) {
              out_contribs[num_feats] += mytree[node].value;
            }

            // Currently assuming a single output
            if (N != 0) {
                if (M != 0) {
                    pos_lst[node] = mytree[node].value * memoized_weights[N + max_depth * (M-1)];
                }
                if (M != N) {
                    neg_lst[node] = -mytree[node].value * memoized_weights[N + max_depth * M];
                }
            }
//             if (DEBUG) {
//               myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//               myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//             }
            // Pop from node_stack
            ns_ctr -= 1;
            next_node = node_stack[ns_ctr];
            from_child = node;
            // Unwind
            if (feat_hist[pfeat] > 0) {
                feat_hist[pfeat] -= 1;
            } else if (feat_hist[pfeat] < 0) {
                feat_hist[pfeat] += 1;
            }
            if (feat_hist[pfeat] == 0) {
                if (from_flag == FROM_X_NOT_R) {
                    N = N-1;
                    M = M-1;
                } else if (from_flag == FROM_R_NOT_X) {
                    N = N-1;
                }
            }
            continue;
        }

        const bool x_right = x[feat] > thres;
        const bool r_right = r[feat] > thres;

        if (x_missing[feat]) {
            next_xnode = cd;
        } else if (x_right) {
            next_xnode = cr;
        } else if (!x_right) {
            next_xnode = cl;
        }
        
        if (r_missing[feat]) {
            next_rnode = cd;
        } else if (r_right) {
            next_rnode = cr;
        } else if (!r_right) {
            next_rnode = cl;
        }

        if (next_xnode >= 0) {
          if (next_xnode != next_rnode) {
              mytree[next_xnode].from_flag = FROM_X_NOT_R;
              mytree[next_rnode].from_flag = FROM_R_NOT_X;
          } else {
              mytree[next_xnode].from_flag = FROM_NEITHER;
          }
        }
        
        // Arriving at node from parent
        if (from_child == -1) {
            //      if (DEBUG) {
            //        myfile << "Arriving at node from parent\n";
            //      }
            node_stack[ns_ctr] = node;
            ns_ctr += 1;
            next_node = -1;
            
            //      if (DEBUG) {
            //        myfile << "feat_hist[feat]" << feat_hist[feat] << "\n";
            //      }
            // Feature is set upstream
            if (feat_hist[feat] > 0) {
                next_node = next_xnode;
                feat_hist[feat] += 1;
            } else if (feat_hist[feat] < 0) {
                next_node = next_rnode;
                feat_hist[feat] -= 1;
            }
            
            // x and r go the same way
            if (next_node < 0) {
                if (next_xnode == next_rnode) {
                    next_node = next_xnode;
                }
            }
            
            // Go down one path
            if (next_node >= 0) {
                continue;
            }
            
            // Go down both paths, but go left first
            next_node = cl;
            if (next_rnode == next_node) {
                N = N+1;
                feat_hist[feat] -= 1;
            } else if (next_xnode == next_node) {
                M = M+1;
                N = N+1;
                feat_hist[feat] += 1;
            }
            from_child = -1;
            continue;
        }
        
        // Arriving at node from child
        if (from_child != -1) {
//             if (DEBUG) {
//               myfile << "Arriving at node from child\n";
//             }
            next_node = -1;
            // Check if we should unroll immediately
            if ((next_rnode == next_xnode) || (feat_hist[feat] != 0)) {
                next_node = pnode;
            }
            
            // Came from a single path, so unroll
            if (next_node >= 0) {
//                 if (DEBUG) {
//                   myfile << "Came from a single path, so unroll\n";
//                 }
                // At the root node
                if (node == 0) {
                    break;
                }
                // Update and unroll
                pos_lst[node] = pos_lst[from_child];
                neg_lst[node] = neg_lst[from_child];

//                 if (DEBUG) {
//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//                 }
                from_child = node;
                ns_ctr -= 1;
                
                // Unwind
                if (feat_hist[pfeat] > 0) {
                    feat_hist[pfeat] -= 1;
                } else if (feat_hist[pfeat] < 0) {
                    feat_hist[pfeat] += 1;
                }
                if (feat_hist[pfeat] == 0) {
                    if (from_flag == FROM_X_NOT_R) {
                        N = N-1;
                        M = M-1;
                    } else if (from_flag == FROM_R_NOT_X) {
                        N = N-1;
                    }
                }
                continue;
                // Go right - Arriving from the left child
            } else if (from_child == cl) {
//                 if (DEBUG) {
//                   myfile << "Go right - Arriving from the left child\n";
//                 }
                node_stack[ns_ctr] = node;
                ns_ctr += 1;
                next_node = cr;
                if (next_xnode == next_node) {
                    M = M+1;
                    N = N+1;
                    feat_hist[feat] += 1;
                } else if (next_rnode == next_node) {
                    N = N+1;
                    feat_hist[feat] -= 1;
                }
                from_child = -1;
                continue;
                // Compute stuff and unroll - Arriving from the right child
            } else if (from_child == cr) {
//                 if (DEBUG) {
//                   myfile << "Compute stuff and unroll - Arriving from the right child\n";
//                 }
                pos_x = 0;
                neg_x = 0;
                pos_r = 0;
                neg_r = 0;
                if ((next_xnode == cr) && (next_rnode == cl)) {
                    pos_x = pos_lst[cr];
                    neg_x = neg_lst[cr];
                    pos_r = pos_lst[cl];
                    neg_r = neg_lst[cl];
                } else if ((next_xnode == cl) && (next_rnode == cr)) {
                    pos_x = pos_lst[cl];
                    neg_x = neg_lst[cl];
                    pos_r = pos_lst[cr];
                    neg_r = neg_lst[cr];
                }
                // out_contribs needs to have been initialized as all zeros
                // if (pos_x + neg_r != 0) {
                //   std::cout << "val " << pos_x + neg_r << "\n";
                // }
                out_contribs[feat] += pos_x + neg_r;
                pos_lst[node] = pos_x + pos_r;
                neg_lst[node] = neg_x + neg_r;

//                 if (DEBUG) {
//                   myfile << "out_contribs[feat]: " << out_contribs[feat] << "\n";
//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//                 }
                
                // Check if at root
                if (node == 0) {
                    break;
                }
                
                // Pop
                ns_ctr -= 1;
                next_node = node_stack[ns_ctr];
                from_child = node;
                
                // Unwind
                if (feat_hist[pfeat] > 0) {
                    feat_hist[pfeat] -= 1;
                } else if (feat_hist[pfeat] < 0) {
                    feat_hist[pfeat] += 1;
                }
                if (feat_hist[pfeat] == 0) {
                    if (from_flag == FROM_X_NOT_R) {
                        N = N-1;
                        M = M-1;
                    } else if (from_flag == FROM_R_NOT_X) {
                        N = N-1;
                    }
                }
                continue;
            }
        }
    }
    //  if (DEBUG) {
    //    myfile.close();
    //  }
}


inline void print_progress_bar(tfloat &last_print, tfloat start_time, unsigned i, unsigned total_count) {
    const tfloat elapsed_seconds = difftime(time(NULL), start_time);
    
    if (elapsed_seconds > 10 && elapsed_seconds - last_print > 0.5) {
        const tfloat fraction = static_cast<tfloat>(i) / total_count;
        const double total_seconds = elapsed_seconds / fraction;
        last_print = elapsed_seconds;
        
        PySys_WriteStderr(
            "\r%3.0f%%|%.*s%.*s| %d/%d [%02d:%02d<%02d:%02d]       ",
            fraction * 100, int(0.5 + fraction*20), "===================",
            20-int(0.5 + fraction*20), "                   ",
            i, total_count,
            int(elapsed_seconds/60), int(elapsed_seconds) % 60,
            int((total_seconds - elapsed_seconds)/60), int(total_seconds - elapsed_seconds) % 60
        );

        // Get handle to python stderr file and flush it (https://mail.python.org/pipermail/python-list/2004-November/294912.html)
        PyObject *pyStderr = PySys_GetObject("stderr");
        if (pyStderr) {
            PyObject *result = PyObject_CallMethod(pyStderr, "flush", NULL);
            Py_XDECREF(result);
        }
    }
}

/**
 * Runs Tree SHAP with feature independence assumptions on dense data.
 */
inline void dense_independent(const TreeEnsemble& trees, const ExplanationDataset &data,
                       tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {

    // reformat the trees for faster access
    Node *node_trees = new Node[trees.tree_limit * trees.max_nodes];
    for (unsigned i = 0; i < trees.tree_limit; ++i) {
        Node *node_tree = node_trees + i * trees.max_nodes;
        for (unsigned j = 0; j < trees.max_nodes; ++j) {
            const unsigned en_ind = i * trees.max_nodes + j;
            node_tree[j].cl = trees.children_left[en_ind];
            node_tree[j].cr = trees.children_right[en_ind];
            node_tree[j].cd = trees.children_default[en_ind];
            if (j == 0) {
                node_tree[j].pnode = 0;
            }
            if (trees.children_left[en_ind] >= 0) { // relies on all unused entries having negative values in them
                node_tree[trees.children_left[en_ind]].pnode = j;
                node_tree[trees.children_left[en_ind]].pfeat = trees.features[en_ind];
            }
            if (trees.children_right[en_ind] >= 0) { // relies on all unused entries having negative values in them
                node_tree[trees.children_right[en_ind]].pnode = j;
                node_tree[trees.children_right[en_ind]].pfeat = trees.features[en_ind];
            }

            node_tree[j].thres = trees.thresholds[en_ind];
            node_tree[j].feat = trees.features[en_ind];
        }
    }

    // preallocate arrays needed by the algorithm
    float *pos_lst = new float[trees.max_nodes];
    float *neg_lst = new float[trees.max_nodes];
    int *node_stack = new int[(unsigned) trees.max_depth];
    signed short *feat_hist = new signed short[data.M];
    tfloat *tmp_out_contribs = new tfloat[(data.M + 1)];

    // precompute all the weight coefficients
    float *memoized_weights = new float[(trees.max_depth+1) * (trees.max_depth+1)];
    for (unsigned n = 0; n <= trees.max_depth; ++n) {
        for (unsigned m = 0; m <= trees.max_depth; ++m) {
            memoized_weights[n + trees.max_depth * m] = 1.0 / (n * bin_coeff(n-1, m));
        }
    }

    // compute the explanations for each sample
    tfloat *instance_out_contribs;
    tfloat rescale_factor = 1.0;
    tfloat margin_x = 0;
    tfloat margin_r = 0;
    time_t start_time = time(NULL);
    tfloat last_print = 0;
    for (unsigned oind = 0; oind < trees.num_outputs; ++oind) {
        // set the values int he reformated tree to the current output index
        for (unsigned i = 0; i < trees.tree_limit; ++i) {
            Node *node_tree = node_trees + i * trees.max_nodes;
            for (unsigned j = 0; j < trees.max_nodes; ++j) {
                const unsigned en_ind = i * trees.max_nodes + j;
                node_tree[j].value = trees.values[en_ind * trees.num_outputs + oind];
            }
        }

        // loop over all the samples
        for (unsigned i = 0; i < data.num_X; ++i) {
            const tfloat *x = data.X + i * data.M;
            const bool *x_missing = data.X_missing + i * data.M;
            instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
            const tfloat y_i = data.y == NULL ? 0 : data.y[i];

            print_progress_bar(last_print, start_time, oind * data.num_X + i, data.num_X * trees.num_outputs);

            // compute the model's margin output for x
            if (transform != NULL) {
                margin_x = trees.base_offset[oind];
                for (unsigned k = 0; k < trees.tree_limit; ++k) {
                    margin_x += tree_predict(k, trees, x, x_missing)[oind];
                }
            }

            for (unsigned j = 0; j < data.num_R; ++j) {
                const tfloat *r = data.R + j * data.M;
                const bool *r_missing = data.R_missing + j * data.M;
                std::fill_n(tmp_out_contribs, (data.M + 1), 0);

                // compute the model's margin output for r
                if (transform != NULL) {
                    margin_r = trees.base_offset[oind];
                    for (unsigned k = 0; k < trees.tree_limit; ++k) {
                        margin_r += tree_predict(k, trees, r, r_missing)[oind];
                    }
                }

                for (unsigned k = 0; k < trees.tree_limit; ++k) {
                    tree_shap_indep(
                        trees.max_depth, data.M, trees.max_nodes, x, x_missing, r, r_missing, 
                        tmp_out_contribs, pos_lst, neg_lst, feat_hist, memoized_weights, 
                        node_stack, node_trees + k * trees.max_nodes
                    );
                }

                // compute the rescale factor
                if (transform != NULL) {
                    if (margin_x == margin_r) {
                        rescale_factor = 1.0;
                    } else {
                        rescale_factor = (*transform)(margin_x, y_i) - (*transform)(margin_r, y_i);
                        rescale_factor /= margin_x - margin_r;
                    }
                }

                // add the effect of the current reference to our running total
                // this is where we can do per reference scaling for non-linear transformations
                for (unsigned k = 0; k < data.M; ++k) {
                    instance_out_contribs[k * trees.num_outputs + oind] += tmp_out_contribs[k] * rescale_factor;
                }

                // Add the base offset
                if (transform != NULL) {
                    instance_out_contribs[data.M * trees.num_outputs + oind] += (*transform)(trees.base_offset[oind] + tmp_out_contribs[data.M], 0);
                } else {
                    instance_out_contribs[data.M * trees.num_outputs + oind] += trees.base_offset[oind] + tmp_out_contribs[data.M];
                }
            }

            // average the results over all the references.
            for (unsigned j = 0; j < (data.M + 1); ++j) {
                instance_out_contribs[j * trees.num_outputs + oind] /= data.num_R;
            }

            // apply the base offset to the bias term
            // for (unsigned j = 0; j < trees.num_outputs; ++j) {
            //     instance_out_contribs[data.M * trees.num_outputs + j] += (*transform)(trees.base_offset[j], 0);
            // }
        }
    }

    delete[] tmp_out_contribs;
    delete[] node_trees;
    delete[] pos_lst;
    delete[] neg_lst;
    delete[] node_stack;
    delete[] feat_hist;
    delete[] memoized_weights;
}


/**
 * This calculates array for distributing threads evenly across trees (in terms of tree size) for algorithm v2
 */
inline void tree_thread_v2(int *tree_thread, const unsigned int n_jobs, const unsigned int tree_limit) {
    unsigned t = 0;
    for (unsigned i = 0; i < n_jobs; ++i) {
        unsigned j = i;
        while (j < tree_limit) {
            tree_thread[t] = j;
            j += n_jobs;
            t++;
        }
    }
}


/**
 * This runs Tree SHAP with a per tree path conditional dependence assumption.
 */
inline void dense_tree_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
                               tfloat *out_contribs, tfloat transform(const tfloat, const tfloat), const int algorithm, const int n_jobs) {
    tfloat *instance_out_contribs;
    TreeEnsemble tree;
    ExplanationDataset instance;

    // pre-define variables for algorithm v2
    const unsigned max_leaves = (trees.max_nodes + 1) / 2;
    const unsigned max_combinations = static_cast<int>(1 << trees.max_depth);
    tfloat *combination_sum;
    int *duplicated_node;
    int *tree_thread;

    // dispatch to the correct algorithm version
    switch (algorithm) {
        case ALGORITHM::v0:
            // build explanation for each sample
            #pragma omp parallel for private(instance_out_contribs, tree, instance) num_threads(n_jobs)
            for (unsigned i = 0; i < data.num_X; ++i) {
                instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
                data.get_x_instance(instance, i);

                // aggregate the effect of explaining each tree
                // (this works because of the linearity property of Shapley values)
                for (unsigned j = 0; j < trees.tree_limit; ++j) {
                    trees.get_tree(tree, j);
                    tree_shap(tree, instance, instance_out_contribs, 0, 0);
                }
            }
            return;

        case ALGORITHM::v1:
            // build explanation for each sample
            #pragma omp parallel for private(instance_out_contribs, tree, instance) num_threads(n_jobs)
            for (unsigned i = 0; i < data.num_X; ++i) {
                instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
                data.get_x_instance(instance, i);

                // aggregate the effect of explaining each tree
                // (this works because of the linearity property of Shapley values)
                for (unsigned j = 0; j < trees.tree_limit; ++j) {
                    trees.get_tree(tree, j);
                    tree_shap_v1(tree, instance, instance_out_contribs, 0, 0);
                }
            }
            return;

        case ALGORITHM::v2_1:
            // pre-define variables for parallel computing
            tfloat *out_contribs_local;
            tfloat *instance_out_contribs_local;
            
            // array for distributing threads evenly across trees
            tree_thread = new int[trees.tree_limit];
            tree_thread_v2(tree_thread, n_jobs, trees.tree_limit);

            // compute combination sum for each tree and aggregate the effect of explaining each tree
            // (this works because of the linearity property of Shapley values)
            #pragma omp parallel private(instance_out_contribs, tree, instance, combination_sum, duplicated_node, \
            out_contribs_local, instance_out_contribs_local) num_threads(n_jobs)
            {
                out_contribs_local = new tfloat[data.num_X * (data.M + 1) * trees.num_outputs];
                for (unsigned i = 0; i < data.num_X; ++i) {
                    instance_out_contribs_local = out_contribs_local + i * (data.M + 1) * trees.num_outputs;
                    for (unsigned k = 0; k < (data.M + 1) * trees.num_outputs; ++k) {
                        instance_out_contribs_local[k] = 0;
                    }
                }
                combination_sum = new tfloat[max_leaves * max_combinations];
                duplicated_node = new int[trees.max_nodes];
                
                #pragma omp for schedule(dynamic)
                for (unsigned j = 0; j < trees.tree_limit; ++j) {
                    trees.get_tree(tree, tree_thread[j]);
                    compute_combination_sum_v2(tree, combination_sum, duplicated_node);

                    // build explanation for each sample
                    for (unsigned i = 0; i < data.num_X; ++i) {
                        instance_out_contribs_local = out_contribs_local + i * (data.M + 1) * trees.num_outputs;
                        data.get_x_instance(instance, i);
                        tree_shap_v2(tree, combination_sum, duplicated_node, instance, instance_out_contribs_local);
                    }
                }
                delete[] combination_sum;
                delete[] duplicated_node;

                #pragma omp critical
                for (unsigned i = 0; i < data.num_X; ++i) {
                    instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
                    instance_out_contribs_local = out_contribs_local + i * (data.M + 1) * trees.num_outputs;
                    for (unsigned k = 0; k < (data.M + 1) * trees.num_outputs; ++k) {
                        instance_out_contribs[k] += instance_out_contribs_local[k];
                    }
                }
                delete[] out_contribs_local;
            }
            delete[] tree_thread;
            return;

        case ALGORITHM::v2_2:
            // pre-allocate space for combination sum and duplicated node
            combination_sum = new tfloat[max_leaves * max_combinations * trees.tree_limit];
            duplicated_node = new int[trees.max_nodes * trees.tree_limit];

            // pre-define variables for parallel computing
            tfloat *combination_sum_local;
            int *duplicated_node_local;

            // array for distributing threads evenly across trees
            tree_thread = new int[trees.tree_limit];
            tree_thread_v2(tree_thread, n_jobs, trees.tree_limit);

            // compute combination sum for each tree
            #pragma omp parallel private(tree, combination_sum_local, duplicated_node_local) num_threads(n_jobs)
            {
                #pragma omp for schedule(dynamic)
                for (unsigned j = 0; j < trees.tree_limit; ++j) {
                    combination_sum_local = combination_sum + tree_thread[j] * max_leaves * max_combinations;
                    duplicated_node_local = duplicated_node + tree_thread[j] * trees.max_nodes;
                    trees.get_tree(tree, tree_thread[j]);
                    compute_combination_sum_v2(tree, combination_sum_local, duplicated_node_local);
                }
            }

            // aggregate the effect of explaining each tree
            // (this works because of the linearity property of Shapley values)
            #pragma omp parallel private(instance_out_contribs, tree, instance, combination_sum_local, duplicated_node_local) num_threads(n_jobs)
            {
                #pragma omp for
                for (unsigned i = 0; i < data.num_X; ++i) {
                    instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
                    data.get_x_instance(instance, i);
                    for (unsigned j = 0; j < trees.tree_limit; ++j) {
                        combination_sum_local = combination_sum + j * max_leaves * max_combinations;
                        duplicated_node_local = duplicated_node + j * trees.max_nodes;
                        trees.get_tree(tree, j);
                        tree_shap_v2(tree, combination_sum_local, duplicated_node_local, instance, instance_out_contribs);
                    }
                }
            }
            delete[] combination_sum;
            delete[] duplicated_node;
            delete[] tree_thread;
            return;

        case ALGORITHM::v3: {
            // batched descent: one traversal per tree carrying all samples as vectors.
            // transpose X once so each split reads a contiguous feature column
            tfloat *Xt = new tfloat[(size_t)data.M * data.num_X];
            for (unsigned i = 0; i < data.num_X; ++i) {
                for (unsigned f = 0; f < data.M; ++f) {
                    Xt[(size_t)f * data.num_X + i] = data.X[(size_t)i * data.M + f];
                }
            }

            #pragma omp parallel private(tree) num_threads(n_jobs)
            {
                const unsigned num_X = data.num_X;
                const unsigned phi_width = (data.M + 1) * trees.num_outputs;
                const unsigned levels = trees.max_depth + 2;
                tfloat *phi_t = new tfloat[(size_t)phi_width * num_X];
                for (size_t q = 0; q < (size_t)phi_width * num_X; ++q) phi_t[q] = 0;
                tfloat *combination_sum_v3 = new tfloat[max_leaves * max_combinations];
                int *duplicated_node_v3 = new int[trees.max_nodes];
                unsigned *K = new unsigned[(size_t)levels * num_X];
                tfloat *R = new tfloat[(size_t)levels * num_X];
                unsigned char *B = new unsigned char[(size_t)levels * num_X];
                unsigned char *INC = new unsigned char[(size_t)levels * num_X];
                BatchedPathElement *path = new BatchedPathElement[(levels + 1) * (levels + 2) / 2];
                int leaf_count;

                #pragma omp for schedule(dynamic)
                for (unsigned j = 0; j < trees.tree_limit; ++j) {
                    trees.get_tree(tree, j);
                    compute_combination_sum_v2(tree, combination_sum_v3, duplicated_node_v3);

                    // expected value of this tree goes to the bias column of every sample
                    for (unsigned jo = 0; jo < tree.num_outputs; ++jo) {
                        tfloat *bias_row = phi_t + ((size_t)data.M * tree.num_outputs + jo) * num_X;
                        const tfloat ev = tree.values[jo];
                        for (unsigned m = 0; m < num_X; ++m) bias_row[m] += ev;
                    }

                    TreeShapV3Context ctx;
                    ctx.num_X = num_X;
                    ctx.num_outputs = tree.num_outputs;
                    ctx.children_left = tree.children_left;
                    ctx.children_right = tree.children_right;
                    ctx.features = tree.features;
                    ctx.thresholds = tree.thresholds;
                    ctx.values = tree.values;
                    ctx.node_sample_weight = tree.node_sample_weights;
                    ctx.max_depth = tree.max_depth;
                    ctx.combination_sum = combination_sum_v3;
                    ctx.duplicated_node = duplicated_node_v3;
                    ctx.Xt = Xt;
                    ctx.phi_t = phi_t;
                    ctx.K = K; ctx.R = R; ctx.B = B; ctx.INC = INC;
                    ctx.path = path;
                    leaf_count = 0;
                    ctx.leaf_count = &leaf_count;
                    tree_shap_recursive_v3(ctx, 0, 0, 0, path, 1, -1);
                }

                // merge the transposed thread-local phi into sample-major out_contribs
                #pragma omp critical
                for (unsigned i = 0; i < num_X; ++i) {
                    tfloat *oc = out_contribs + (size_t)i * phi_width;
                    for (unsigned c = 0; c < phi_width; ++c) {
                        oc[c] += phi_t[(size_t)c * num_X + i];
                    }
                }
                delete[] phi_t;
                delete[] combination_sum_v3;
                delete[] duplicated_node_v3;
                delete[] K; delete[] R; delete[] B; delete[] INC;
                delete[] path;
            }
            delete[] Xt;
            return;
        }
    }

    // apply the base offset to the bias term
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset[j];
        }
    }
}

// phi = np.zeros((self._current_X.shape[1] + 1, self._current_X.shape[1] + 1, self.n_outputs))
//         phi_diag = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
//         for t in range(self.tree_limit):
//             self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_diag)
//             for j in self.trees[t].unique_features:
//                 phi_on = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
//                 phi_off = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
//                 self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_on, 1, j)
//                 self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_off, -1, j)
//                 phi[j] += np.true_divide(np.subtract(phi_on,phi_off),2.0)
//                 phi_diag[j] -= np.sum(np.true_divide(np.subtract(phi_on,phi_off),2.0))
//         for j in range(self._current_X.shape[1]+1):
//             phi[j][j] = phi_diag[j]
//         phi /= self.tree_limit
//         return phi

inline void dense_tree_interactions_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
                                            tfloat *out_contribs,
                                            tfloat transform(const tfloat, const tfloat), const int algorithm, const int n_jobs) {

    // build a list of all the unique features in each tree
    int amount_of_unique_features = min(data.M, trees.max_nodes);
    int *unique_features = new int[trees.tree_limit * amount_of_unique_features];
    std::fill(unique_features, unique_features + trees.tree_limit * amount_of_unique_features, -1);
    for (unsigned j = 0; j < trees.tree_limit; ++j) {
        const int *features_row = trees.features + j * trees.max_nodes;
        int *unique_features_row = unique_features + j * amount_of_unique_features;
        for (unsigned k = 0; k < trees.max_nodes; ++k) {
            for (unsigned l = 0; l < amount_of_unique_features; ++l) {
                if (features_row[k] == unique_features_row[l]) break;
                if (unique_features_row[l] < 0) {
                    unique_features_row[l] = features_row[k];
                    break;
                }
            }
        }
    }
    
    // build an interaction explanation for each sample
    tfloat *instance_out_contribs;
    TreeEnsemble tree;
    ExplanationDataset instance;
    const unsigned contrib_row_size = (data.M + 1) * trees.num_outputs;
    tfloat *diag_contribs;
    tfloat *on_contribs;
    tfloat *off_contribs;

    // dispatch to the correct algorithm version
    switch (algorithm) {
        case ALGORITHM::v0:
            #pragma omp parallel private(instance_out_contribs, tree, instance, diag_contribs, on_contribs, off_contribs) num_threads(n_jobs)
            {
                diag_contribs = new tfloat[contrib_row_size];
                on_contribs = new tfloat[contrib_row_size];
                off_contribs = new tfloat[contrib_row_size];

                #pragma omp for
                for (unsigned i = 0; i < data.num_X; ++i) {
                    instance_out_contribs = out_contribs + i * (data.M + 1) * contrib_row_size;
                    data.get_x_instance(instance, i);

                    // aggregate the effect of explaining each tree
                    // (this works because of the linearity property of Shapley values)
                    std::fill(diag_contribs, diag_contribs + contrib_row_size, 0);
                    for (unsigned j = 0; j < trees.tree_limit; ++j) {
                        trees.get_tree(tree, j);
                        tree_shap(tree, instance, diag_contribs, 0, 0);

                        const int *unique_features_row = unique_features + j * amount_of_unique_features;
                        for (unsigned k = 0; k < amount_of_unique_features; ++k) {
                            const int ind = unique_features_row[k];
                            if (ind < 0) break; // < 0 means we have seen all the features for this tree

                            // compute the shap value with this feature held on and off
                            std::fill(on_contribs, on_contribs + contrib_row_size, 0);
                            std::fill(off_contribs, off_contribs + contrib_row_size, 0);
                            tree_shap(tree, instance, on_contribs, 1, ind);
                            tree_shap(tree, instance, off_contribs, -1, ind);

                            // save the difference between on and off as the interaction value
                            for (unsigned l = 0; l < contrib_row_size; ++l) {
                                const tfloat val = (on_contribs[l] - off_contribs[l]) / 2;
                                instance_out_contribs[ind * contrib_row_size + l] += val;
                                diag_contribs[l] -= val;
                            }
                        }
                    }

                    // set the diagonal
                    for (unsigned j = 0; j < data.M + 1; ++j) {
                        const unsigned offset = j * contrib_row_size + j * trees.num_outputs;
                        for (unsigned k = 0; k < trees.num_outputs; ++k) {
                            instance_out_contribs[offset + k] = diag_contribs[j * trees.num_outputs + k];
                        }
                    }

                    // apply the base offset to the bias term
                    const unsigned last_ind = (data.M * (data.M + 1) + data.M) * trees.num_outputs;
                    for (unsigned j = 0; j < trees.num_outputs; ++j) {
                        instance_out_contribs[last_ind + j] += trees.base_offset[j];
                    }
                }
                delete[] diag_contribs;
                delete[] on_contribs;
                delete[] off_contribs;
            }
            return;

        case ALGORITHM::v1:
            #pragma omp parallel private(instance_out_contribs, tree, instance, diag_contribs, on_contribs, off_contribs) num_threads(n_jobs)
            {
                diag_contribs = new tfloat[contrib_row_size];
                on_contribs = new tfloat[contrib_row_size];
                off_contribs = new tfloat[contrib_row_size];

                #pragma omp for
                for (unsigned i = 0; i < data.num_X; ++i) {
                    instance_out_contribs = out_contribs + i * (data.M + 1) * contrib_row_size;
                    data.get_x_instance(instance, i);

                    // aggregate the effect of explaining each tree
                    // (this works because of the linearity property of Shapley values)
                    std::fill(diag_contribs, diag_contribs + contrib_row_size, 0);
                    for (unsigned j = 0; j < trees.tree_limit; ++j) {
                        trees.get_tree(tree, j);
                        tree_shap_v1(tree, instance, diag_contribs, 0, 0);

                        const int *unique_features_row = unique_features + j * amount_of_unique_features;
                        for (unsigned k = 0; k < amount_of_unique_features; ++k) {
                            const int ind = unique_features_row[k];
                            if (ind < 0) break; // < 0 means we have seen all the features for this tree

                            // compute the shap value with this feature held on and off
                            std::fill(on_contribs, on_contribs + contrib_row_size, 0);
                            std::fill(off_contribs, off_contribs + contrib_row_size, 0);
                            tree_shap_v1(tree, instance, on_contribs, 1, ind);
                            tree_shap_v1(tree, instance, off_contribs, -1, ind);

                            // save the difference between on and off as the interaction value
                            for (unsigned l = 0; l < contrib_row_size; ++l) {
                                const tfloat val = (on_contribs[l] - off_contribs[l]) / 2;
                                instance_out_contribs[ind * contrib_row_size + l] += val;
                                diag_contribs[l] -= val;
                            }
                        }
                    }

                    // set the diagonal
                    for (unsigned j = 0; j < data.M + 1; ++j) {
                        const unsigned offset = j * contrib_row_size + j * trees.num_outputs;
                        for (unsigned k = 0; k < trees.num_outputs; ++k) {
                            instance_out_contribs[offset + k] = diag_contribs[j * trees.num_outputs + k];
                        }
                    }

                    // apply the base offset to the bias term
                    const unsigned last_ind = (data.M * (data.M + 1) + data.M) * trees.num_outputs;
                    for (unsigned j = 0; j < trees.num_outputs; ++j) {
                        instance_out_contribs[last_ind + j] += trees.base_offset[j];
                    }
                }
                delete[] diag_contribs;
                delete[] on_contribs;
                delete[] off_contribs;
            }
            return;

        case ALGORITHM::v2_1:
            std::cerr << "ALGORITHM::v2 does not support interactions!\n";
            return;

        case ALGORITHM::v2_2:
            std::cerr << "ALGORITHM::v2 does not support interactions!\n";
            return;
    }

    delete[] unique_features;
}

/**
 * This runs Tree SHAP with a global path conditional dependence assumption.
 * 
 * By first merging all the trees in a tree ensemble into an equivalent single tree
 * this method allows arbitrary marginal transformations and also ensures that all the
 * evaluations of the model are consistent with some training data point.
 */
inline void dense_global_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
                                 tfloat *out_contribs, tfloat transform(const tfloat, const tfloat)) {

    // allocate space for our new merged tree (we save enough room to totally split all samples if need be)
    TreeEnsemble merged_tree;
    merged_tree.allocate(1, (data.num_X + data.num_R) * 2, trees.num_outputs);
    
    // collapse the ensemble of trees into a single tree that has the same behavior
    // for all the X and R samples in the dataset
    build_merged_tree(merged_tree, data, trees);

    // compute the expected value and depth of the new merged tree
    compute_expectations(merged_tree);

    // explain each sample using our new merged tree
    ExplanationDataset instance;
    tfloat *instance_out_contribs;
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
        data.get_x_instance(instance, i);
       
        // since we now just have a single merged tree we can just use the tree_path_dependent algorithm
        tree_shap(merged_tree, instance, instance_out_contribs, 0, 0);

        // apply the base offset to the bias term
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset[j];
        }
    }

    merged_tree.free();
}


/**
 * The main method for computing Tree SHAP on models using dense data.
 */
inline void dense_tree_shap(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
                     const int feature_dependence, unsigned model_transform, const int algorithm, const int n_jobs, bool interactions) {

    // see what transform (if any) we have
    transform_f transform = get_transform(model_transform);

    // dispatch to the correct algorithm handler
    switch (feature_dependence) {
        case FEATURE_DEPENDENCE::independent:
            if (interactions) {
                std::cerr << "FEATURE_DEPENDENCE::independent does not support interactions!\n";
            } else dense_independent(trees, data, out_contribs, transform);
            return;
        
        case FEATURE_DEPENDENCE::tree_path_dependent:
            if (interactions) dense_tree_interactions_path_dependent(trees, data, out_contribs, transform, algorithm, n_jobs);
            else dense_tree_path_dependent(trees, data, out_contribs, transform, algorithm, n_jobs);
            return;

        case FEATURE_DEPENDENCE::global_path_dependent:
            if (interactions) {
                std::cerr << "FEATURE_DEPENDENCE::global_path_dependent does not support interactions!\n";
            } else dense_global_path_dependent(trees, data, out_contribs, transform);
            return;
    }
}
