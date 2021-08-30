(define (problem rovers-2-2-rover0) (:domain rovers-2-2-rover0)
(:objects
    base - waypoint
    quarry - waypoint
    dune - waypoint
    mothership - lander
    rover0RockStore - store
    rover0soilStore - store
    rover1RockStore - store
    rover1soilStore - store


	(:private
		rover0 - rover
	)
)
(:init

    ;; private
	(store_of rover0RockStore rover0)
	(store_of rover0soilStore rover0)
	(equipped_for_rock_analysis rover0)
	(equipped_for_soil_analysis rover0)
	(can_traverse rover0 base quarry)
	(can_traverse rover0 quarry base)
	(can_traverse rover0 base dune)
	(can_traverse rover0 dune base)
	(at rover0 base)

	;; public
	(visible quarry base)
	(visible base quarry)
	(visible dune base)
	(visible base dune)
	(at_soil_sample dune)
	(at_rock_sample quarry)
	(at_lander mothership base)
	(empty rover0RockStore)
	(empty rover0soilStore)
	(empty rover1RockStore)
	(empty rover1soilStore)
)
(:goal
	(and
		(communicated_rock_data quarry)
		(communicated_soil_data dune)
	)
)
)