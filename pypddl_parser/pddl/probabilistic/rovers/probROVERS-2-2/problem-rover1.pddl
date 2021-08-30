(define (problem rovers-2-2-rover1) (:domain rovers-2-2-rover1)
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
		rover1 - rover
	)
)
(:init

    ;; private
	(store_of rover1RockStore rover1)
	(store_of rover1soilStore rover1)
	(equipped_for_rock_analysis rover1)
	(equipped_for_soil_analysis rover1)
	(can_traverse rover1 base quarry)
	(can_traverse rover1 quarry base)
	(can_traverse rover1 base dune)
	(can_traverse rover1 dune base)
	(at rover1 base)

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