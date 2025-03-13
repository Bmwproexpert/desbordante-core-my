#pragma once
#include "algorithm.h"
#include "algorithms/dd/dd.h"
#include "table/column_layout_relation_data.h"
#include "table/column_layout_typed_relation_data.h"
#include "tabular_data/input_table_type.h"

namespace algos::dd {
using DFs = model::DFStringConstraint;
using DDs = model::DDString;

class DDVerifier : public Algorithm {
private:
    DDs dd_;
    config::InputTable input_table_;
    std::size_t num_rows_{};
    std::size_t num_columns_{};
    std::size_t num_error_rhs_{};
    double error_ = 0.;
    std::shared_ptr<ColumnLayoutRelationData> relation_;
    std::shared_ptr<model::ColumnLayoutTypedRelationData> typed_relation_;
    std::vector<std::pair<std::size_t, std::pair<int, int> > > highlights_;

    void RegisterOptions();

    void VisualizeHighlights();

    void PrintStatistics();

    std::vector<std::pair<int, int> > GetRowsWhereLhsHolds(
            std::list<model::DFStringConstraint> const &constraints) const;

    double CalculateDistance(model::ColumnIndex column_index,
                             std::pair<std::size_t, std::size_t> const &tuple_pair) const;

    void CheckDFOnRhs(std::vector<std::pair<int, int> > const &lhs);

    void ResetState() final {
        num_columns_ = 0;
        num_rows_ = 0;
    }

protected:
    void LoadDataInternal() override;

    void MakeExecuteOptsAvailable() override;

    unsigned long long ExecuteInternal() override;

public:
    DDVerifier();

    double GetError() const;

    std::size_t GetNumErrorRhs() const;

    bool DDHolds() const;

    void VerifyDD();

    std::vector<std::pair<std::size_t, std::pair<int, int> > > GetHighlights() const;
};
}  // namespace algos::dd
