#pragma once

#include <cstddef>
#include <filesystem>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "algorithms/algorithm.h"
#include "algorithms/dd/dd.h"
#include "config/tabular_data/input_table_type.h"
#include "enums.h"
#include "model/table/column_index.h"
#include "model/table/column_layout_relation_data.h"
#include "model/table/column_layout_typed_relation_data.h"

namespace algos::dd {

using DF = model::DF;
using DDString = model::DD;

class Split : public Algorithm {
private:
    config::InputTable input_table_;

    std::shared_ptr<ColumnLayoutRelationData> relation_;
    std::shared_ptr<model::ColumnLayoutTypedRelationData> typed_relation_;
    unsigned num_rows_;
    model::ColumnIndex num_columns_;

    bool has_dif_table_;

    config::InputTable difference_table_;
    std::unique_ptr<model::ColumnLayoutTypedRelationData> difference_typed_relation_;

    Reduce const reduce_method_ = Reduce::IEHybrid;  // currently, the fastest method
    unsigned const num_dfs_per_column_ = 5;

    std::vector<model::DFConstraint> min_max_dif_;
    std::vector<std::vector<std::vector<double>>> distances_;
    std::vector<std::pair<std::size_t, std::size_t>> tuple_pairs_;
    std::list<DDString> dd_collection_;

    void RegisterOptions();
    void SetLimits();
    void ParseDifferenceTable();

    void ResetState() final {
        dd_collection_.clear();
    }

    double CalculateDistance(model::ColumnIndex column_index,
                             std::pair<std::size_t, std::size_t> tuple_pair);
    void InsertDistance(model::ColumnIndex column_index, std::size_t first_index,
                        std::size_t second_index, double& min_dif, double& max_dif);
    bool CheckDF(DF const& dep, std::pair<std::size_t, std::size_t> tuple_pair);
    bool VerifyDD(DDString const& dep);
    void CalculateAllDistances();
    bool IsFeasible(DF const& d);
    std::vector<DF> SearchSpace(std::vector<model::ColumnIndex>& indices);
    std::vector<DF> SearchSpace(model::ColumnIndex index);
    bool Subsume(DF const& df1, DF const& df2);
    std::vector<DF> DoNegativePruning(std::vector<DF> const& search, DF const& last_df);
    std::pair<std::vector<DF>, std::vector<DF>> NegativeSplit(std::vector<DF> const& search,
                                                              DF const& last_df);
    std::vector<DF> DoPositivePruning(std::vector<DF> const& search, DF const& first_df);
    std::pair<std::vector<DF>, std::vector<DF>> PositiveSplit(std::vector<DF> const& search,
                                                              DF const& first_df);
    std::list<DDString> MergeReducedResults(std::list<DDString> const& base_dds,
                                      std::list<DDString> const& dds_to_merge);
    std::list<DDString> NegativePruningReduce(DF const& rhs, std::vector<DF> const& search,
                                        unsigned& cnt);
    std::list<DDString> HybridPruningReduce(DF const& rhs, std::vector<DF> const& search, unsigned& cnt);
    std::list<DDString> InstanceExclusionReduce(
            std::vector<std::pair<std::size_t, std::size_t>> const& tuple_pairs,
            std::vector<DF> const& search, DF const& rhs, unsigned& cnt);
    void CalculateTuplePairs();
    unsigned ReduceDDs(auto const& start_time);
    unsigned RemoveRedundantDDs();
    unsigned RemoveTransitiveDDs();
    model::DDString DDToDDString(DDString const& dd) const;
    void PrintResults();

protected:
    void LoadDataInternal() override;
    void MakeExecuteOptsAvailable() override;
    unsigned long long ExecuteInternal() override;

public:
    Split();
    std::list<DDString> const& GetDDs() const;
    std::vector<model::DFConstraint> const& GetMinMaxDif() const;
    std::list<model::DDString> GetDDStringList() const;
};

}  // namespace algos::dd
