(define (problem depot-8-5-depot0) (:domain depot-8-5-depot0)
(:objects
	truck1 - truck
	truck0 - truck
	depot0 - depot
	crate2 - crate
	crate1 - crate
	crate0 - crate
	pallet1 - pallet
	pallet0 - pallet
	pallet2 - pallet
	distributor1 - distributor
	distributor0 - distributor

	(:private
		hoist0 - hoist
	)
)
(:init
	(available depot0 hoist0)
	(at hoist0 depot0)
	(at truck0 distributor1)
	(at truck1 depot0)

	(at pallet0 depot0)
	(at pallet1 distributor0)
	(at pallet2 distributor1)

	(at crate0 distributor1)
	(at crate1 depot0)
	(at crate2 distributor0)

	(on crate0 pallet2)
	(on crate1 pallet0)
	(on crate2 pallet1)

	(clear crate2)
	(clear crate1)
	(clear crate0)
)
(:goal
	(and
		(on crate0 pallet0)
		(on crate1 pallet1)
		(on crate2 pallet2)
	)
)
)